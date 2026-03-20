from dataclasses import dataclass
from typing import Iterable, Self, Generator

from django.contrib.auth import get_user_model
from django.db import transaction

from .models import Chat as ChatDBModel, Project
from .models import LLMCall, Message
from .services import LLMCacheService
from .backends import LiteLLMProvider, LMStudioProvider


class DuplicateSystemMessageError(Exception):
    pass


User = get_user_model()


@dataclass
class Chat:
    chat_db_model: ChatDBModel
    llm_user: object
    last_user_message: Message | None = None
    llm_call: LLMCall | None = None
    last_llm_message: Message | None = None
    system_message: Message | None = None

    @classmethod
    def create(cls, project: Project | None = None) -> Self:
        try:
            llm_user = ChatDBModel.get_llm_user()
        except User.DoesNotExist:
            llm_user = ChatDBModel.create_llm_user()

        db_model = ChatDBModel.objects.create(project=project)
        return cls(db_model, llm_user)

    def create_user_message(self, text: str, user: object) -> Message:
        return Message.create_user_message(
            chat=self.chat_db_model,
            text=text,
            user=user,
        )

    def get_msg_history(self) -> Iterable[Message]:
        return self.chat_db_model.get_messages().all()

    def create_llm_call(self, *messages: Iterable[Message]) -> LLMCall:
        return LLMCall.create(*messages)

    def add_tokens(self, input_token_count: int, output_token_count: int):
        self.chat_db_model.add_token_counts(input_token_count, output_token_count)

    def _get_backend_provider(self, model_name: str):
        if model_name.startswith("lm_studio"):
            return LMStudioProvider()
        return LiteLLMProvider()

    @transaction.atomic
    def call_llm(
        self,
        model_name,
        message: str,  # TODO: in README mention this is the only thing you need if you send only one message
        user: object,
        include_chat_history: bool = True,
        use_cache: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        user_msg: Message = self.create_user_message(message, user)
        messages = list(self.get_msg_history()) if include_chat_history else [user_msg]
        llm_call = self.create_llm_call(*messages)

        cache_key = None
        if use_cache:
            cache_key = LLMCacheService.compute_cache_key(
                model_name, messages, temperature=temperature, max_tokens=max_tokens
            )
            cache_item = LLMCacheService.lookup_cache(cache_key)
            if cache_item:
                response_text, response_data = (
                    cache_item.response_text,
                    cache_item.response_data.copy(),
                )
                response_data["usage"] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
            else:
                provider = self._get_backend_provider(model_name)
                response_text, response_data = provider.generate(
                    model_name, messages, temperature=temperature, max_tokens=max_tokens
                )
                LLMCacheService.save_to_cache(
                    cache_key, model_name, response_text, response_data
                )
        else:
            provider = self._get_backend_provider(model_name)
            response_text, response_data = provider.generate(
                model_name, messages, temperature=temperature, max_tokens=max_tokens
            )

        input_tokens = response_data["usage"]["prompt_tokens"]
        output_tokens = response_data["usage"]["completion_tokens"]

        self.add_tokens(input_tokens, output_tokens)
        llm_call.add_response_data(response_data, input_tokens, output_tokens)

        llm_msg = Message.create_llm_message(
            user=self.llm_user, text=response_text, chat=self.chat_db_model
        )
        llm_call.add_message(llm_msg)

        self.last_user_message = user_msg
        self.llm_call = llm_call
        self.last_llm_message = llm_msg
        self.system_message = self.chat_db_model.messages.filter(
            type=Message.Type.SYSTEM
        ).first()

    @transaction.atomic
    def stream_call_llm(
        self,
        model_name: str,
        message: str,
        user: object,
        include_chat_history: bool = True,
        use_cache: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Generator[str, None, None]:
        user_msg: Message = self.create_user_message(message, user)
        messages_history = (
            list(self.get_msg_history()) if include_chat_history else [user_msg]
        )
        llm_call = self.create_llm_call(*messages_history)

        self.last_user_message = user_msg
        self.llm_call = llm_call
        self.system_message = self.chat_db_model.messages.filter(
            type=Message.Type.SYSTEM
        ).first()

        cache_key = None
        if use_cache:
            cache_key = LLMCacheService.compute_cache_key(
                model_name,
                messages_history,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            cache_item = LLMCacheService.lookup_cache(cache_key)
            if cache_item:
                response_text, response_data = (
                    cache_item.response_text,
                    cache_item.response_data.copy(),
                )
                response_data["usage"] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
                self.add_tokens(0, 0)
                llm_call.add_response_data(response_data, 0, 0)
                llm_msg = Message.create_llm_message(
                    user=self.llm_user, text=response_text, chat=self.chat_db_model
                )
                llm_call.add_message(llm_msg)
                self.last_llm_message = llm_msg
                yield response_text
                return

        provider = self._get_backend_provider(model_name)
        response_text = ""
        response_data = {}

        for part in provider.stream(
            model_name, messages_history, temperature=temperature, max_tokens=max_tokens
        ):
            if isinstance(part, tuple):
                response_text, response_data = part
            else:
                yield part

        if cache_key:
            LLMCacheService.save_to_cache(
                cache_key, model_name, response_text, response_data
            )

        input_tokens = response_data["usage"]["prompt_tokens"]
        output_tokens = response_data["usage"]["completion_tokens"]

        self.add_tokens(input_tokens, output_tokens)
        llm_call.add_response_data(response_data, input_tokens, output_tokens)

        llm_msg = Message.create_llm_message(
            user=self.llm_user, text=response_text, chat=self.chat_db_model
        )
        llm_call.add_message(llm_msg)
        self.last_llm_message = llm_msg

    def create_system_message(self, text: str, user: object) -> Message:
        if self.chat_db_model.messages.filter(type=Message.Type.SYSTEM).exists():
            raise DuplicateSystemMessageError("System message already exists")

        self.system_message = Message.create_system_message(
            self.chat_db_model, text, user
        )
        return self.system_message
