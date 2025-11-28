from typing import Iterable, Self
from dataclasses import dataclass
from .models import Chat as ChatDBModel, Message, LLMCall
from litellm import completion
from django.contrib.auth import get_user_model


class DuplicateSystemMessageError(Exception):
    pass


def create_litellm_user():
    return get_user_model().objects.create_user(
        username="djllmchat-user", password="djllmchat-user"
    )


def create_chat_user():
    return get_user_model().objects.create_user(
        username="djllmchat", password="djllmchat"
    )


User = get_user_model()


@dataclass
class Chat:
    chat_db_model: ChatDBModel
    llm_user: object
    default_user: object

    @classmethod
    def create(cls) -> Self:
        try:
            llm_user = User.objects.get(username="litellm")
        except User.DoesNotExist:
            llm_user = create_litellm_user()

        try:
            default_user = User.objects.get(username="djllmchat")
        except User.DoesNotExist:
            default_user = create_chat_user()

        db_model = ChatDBModel.objects.create()
        return cls(db_model, llm_user, default_user)

    def create_user_message(self, text: str, user=None) -> Message:
        if not user:
            user = self.default_user
        return Message.create_user_message(
            chat=self.chat_db_model,
            text=text,
            user=user,
        )

    def get_msg_history(self) -> Iterable[Message]:
        return self.chat_db_model.messages.order_by("date_created").all()

    def create_llm_call(self, *messages: Iterable[Message]) -> LLMCall:
        return LLMCall.create(*messages)

    def call_llm_via_litellm(
        self, model_name: str, *messages: Iterable[Message]
    ) -> tuple[str, dict]:
        litellm_messages = []
        for msg in messages:
            litellm_messages.append({"content": msg.text, "role": msg.type})

        response = completion(
            model=model_name,
            messages=litellm_messages,
        )

        message = response.choices[0].message.to_dict()
        response_text = response.choices[0].message.content
        message.pop("content")
        message.pop("role")

        msg_id = response.id
        model = response.model
        usage = response.usage.to_dict()

        response_data = {
            "message": message,
            "id": msg_id,
            "model": model,
            "usage": usage,
        }
        return response_text, response_data

    def add_tokens(self, input_token_count: int, output_token_count: int):
        self.chat_db_model.add_token_counts(input_token_count, output_token_count)

    def send_user_msg_to_llm(
        self, model_name, text: str, user=None, include_chat_history: bool = True
    ) -> tuple[Message, Message, LLMCall]:
        if not user:
            user = self.default_user

        user_msg: ChatDBModel = self.create_user_message(text, user)

        if include_chat_history:
            messages = self.get_msg_history()
        else:
            messages = [user_msg]

        llm_call = self.create_llm_call(*messages)

        response_text, response_data = self.call_llm_via_litellm(model_name, *messages)

        input_token_count = response_data["usage"]["prompt_tokens"]
        output_token_count = response_data["usage"]["completion_tokens"]

        self.add_tokens(input_token_count, output_token_count)
        llm_call.add_response_data(response_data, input_token_count, output_token_count)

        llm_msg = Message.create_llm_message(
            user=self.llm_user,
            text=response_text,
            chat=self.chat_db_model,
        )
        llm_call.add_message(llm_msg)

        return llm_msg, user_msg, llm_call

    def create_system_message(self, text: str, user=None) -> Message:
        if self.chat_db_model.messages.filter(type=Message.Type.SYSTEM).exists():
            raise DuplicateSystemMessageError("System message already exists")

        if not user:
            user = self.default_user
        return Message.create_system_message(self.chat_db_model, text, user)
