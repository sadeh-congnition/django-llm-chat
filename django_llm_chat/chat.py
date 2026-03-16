import hashlib
import json
import os
from dataclasses import dataclass
from typing import Iterable, Self

import litellm
from django.contrib.auth import get_user_model

from .models import Chat as ChatDBModel
from .models import LLMCache, LLMCall, Message


class DuplicateSystemMessageError(Exception):
    pass


def create_litellm_user():
    return get_user_model().objects.create_user(username="litellm", password="litellm")


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

    def _compute_cache_key(self, model_name: str, messages: Iterable[Message]) -> str:
        msg_data = [{"role": m.type, "content": m.text} for m in messages]
        key_data = {
            "model": model_name,
            "messages": msg_data,
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode("utf-8")).hexdigest()

    def _lookup_cache(self, cache_key: str) -> LLMCache | None:
        try:
            cache_item = LLMCache.objects.get(cache_key=cache_key)
            cache_item.hit_count += 1
            cache_item.save(update_fields=["hit_count"])
            return cache_item
        except LLMCache.DoesNotExist:
            return None

    def _save_to_cache(
        self, cache_key: str, model_name: str, response_text: str, response_data: dict
    ):
        LLMCache.objects.get_or_create(
            cache_key=cache_key,
            defaults={
                "model_name": model_name,
                "response_text": response_text,
                "response_data": response_data,
            },
        )

    def create_llm_call(self, *messages: Iterable[Message]) -> LLMCall:
        return LLMCall.create(*messages)

    def _prepare_litellm_messages(self, messages: Iterable[Message]) -> list[dict]:
        litellm_messages = []
        for msg in messages:
            litellm_messages.append({"content": msg.text, "role": msg.type})
        return litellm_messages

    def _prepare_lmstudio_messages(self, messages: Iterable[Message]) -> list:
        lms_messages = []
        for msg in messages:
            role = msg.type
            if role == Message.Type.SYSTEM:
                role = "system"
            elif role == Message.Type.USER:
                role = "user"
            elif role == Message.Type.ASSISTANT:
                role = "assistant"

            lms_messages.append({"role": role, "content": msg.text})
        return lms_messages

    def call_llm_via_lmstudio(
        self, model_name: str, *messages: Iterable[Message], cache_key: str = None
    ) -> tuple[str, dict]:
        if cache_key:
            cache_item = self._lookup_cache(cache_key)
            if cache_item:
                response_data = cache_item.response_data.copy()
                response_data["usage"] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
                return cache_item.response_text, response_data

        lms_messages = self._prepare_lmstudio_messages(messages)

        base_url = os.environ.get("LM_STUDIO_API_BASE", "http://localhost:1234")
        url = f"{base_url.rstrip('/')}/v1/chat/completions"

        # We'll use the OpenAI-compatible endpoint for non-streaming if possible,
        # but the user specified /api/v1/chat for streaming.
        # Let's stick to /api/v1/chat as requested for consistency if it supports non-streaming,
        # or use /v1/chat/completions if it's more standard for non-streaming.
        # The user example used /api/v1/chat with stream=True.

        api_url = f"{base_url.rstrip('/')}/api/v1/chat"
        payload = {
            "model": model_name,
            "messages": lms_messages,  # /api/v1/chat uses messages or input?
            # Example used "input" and "system_prompt".
        }

        # Redoing based on user example:
        system_msg = next(
            (m.text for m in messages if m.type == Message.Type.SYSTEM), ""
        )
        user_msg = messages[-1].text if messages else ""

        # Actually, if we have history, we might need to combine them or use an endpoint that supports history properly.
        # The user's example:
        # data = {
        #     "model": "...",
        #     "system_prompt": "...",
        #     "input": "...",
        #     "stream": True,
        # }

        data = {
            "model": model_name,
            "system_prompt": system_msg,
            "input": user_msg,
            "stream": False,
        }

        response = requests.post(api_url, json=data)
        response.raise_for_status()
        result = response.json()

        # Based on the user example's 'chat.end' event structure:
        # {"type":"chat.end","result":{"model_instance_id":"...","output":[{"type":"message","content":"..."}],"stats":{...}}}

        # For non-streaming, the response might be different or we might have to simulate it if we want tokens.
        # Usually /api/v1/chat returns the final result if stream=False.

        output_content = ""
        if "output" in result:
            output_content = "".join(
                [
                    o.get("content", "")
                    for o in result["output"]
                    if o.get("type") == "message"
                ]
            )

        stats = result.get("stats", {})
        prompt_tokens = stats.get("input_tokens", 0)
        completion_tokens = stats.get("total_output_tokens", 0)

        response_data = {
            "message": {"content": output_content},
            "id": f"lms-{result.get('response_id', 'unknown')}",
            "model": model_name,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

        if cache_key:
            self._save_to_cache(cache_key, model_name, output_content, response_data)

        return output_content, response_data

    def call_llm_via_litellm(
        self, model_name: str, *messages: Iterable[Message], cache_key: str = None
    ) -> tuple[str, dict]:
        if cache_key:
            cache_item = self._lookup_cache(cache_key)
            if cache_item:
                response_data = cache_item.response_data.copy()
                response_data["usage"] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
                return cache_item.response_text, response_data

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

        if cache_key:
            self._save_to_cache(cache_key, model_name, response_text, response_data)

        return response_text, response_data

    def add_tokens(self, input_token_count: int, output_token_count: int):
        self.chat_db_model.add_token_counts(input_token_count, output_token_count)

    def send_user_msg_to_llm(
        self,
        model_name,
        text: str,
        user=None,
        include_chat_history: bool = True,
        backend: str = "litellm",
        use_cache: bool = False,
    ) -> tuple[Message, Message, LLMCall]:
        if not user:
            user = self.default_user

        user_msg: Message = self.create_user_message(text, user)

        if include_chat_history:
            messages = list(self.get_msg_history())
        else:
            messages = [user_msg]

        llm_call = self.create_llm_call(*messages)

        cache_key = None
        if use_cache:
            cache_key = self._compute_cache_key(model_name, messages)

        if backend == "lmstudio":
            response_text, response_data = self.call_llm_via_lmstudio(
                model_name, *messages, cache_key=cache_key
            )
        else:
            response_text, response_data = self.call_llm_via_litellm(
                model_name, *messages, cache_key=cache_key
            )

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

    def stream_user_msg_to_llm(
        self,
        model_name,
        text: str,
        user=None,
        include_chat_history: bool = True,
        backend: str = "litellm",
        use_cache: bool = False,
    ) -> Iterable[str]:
        if not user:
            user = self.default_user

        user_msg: Message = self.create_user_message(text, user)

        if include_chat_history:
            messages_history = list(self.get_msg_history())
        else:
            messages_history = [user_msg]

        llm_call = self.create_llm_call(*messages_history)

        cache_key = None
        if use_cache:
            cache_key = self._compute_cache_key(model_name, messages_history)
            cache_item = self._lookup_cache(cache_key)
            if cache_item:
                response_text = cache_item.response_text
                response_data = cache_item.response_data.copy()
                response_data["usage"] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }

                self.add_tokens(0, 0)
                llm_call.add_response_data(response_data, 0, 0)

                llm_msg = Message.create_llm_message(
                    user=self.llm_user,
                    text=response_text,
                    chat=self.chat_db_model,
                )
                llm_call.add_message(llm_msg)

                yield response_text
                return

        if backend == "lmstudio":
            base_url = os.environ.get("LM_STUDIO_API_BASE", "http://localhost:1234")
            api_url = f"{base_url.rstrip('/')}/api/v1/chat"

            system_msg = next(
                (m.text for m in messages_history if m.type == Message.Type.SYSTEM), ""
            )
            user_msg_text = next(
                (
                    m.text
                    for m in reversed(messages_history)
                    if m.type == Message.Type.USER
                ),
                "",
            )

            data = {
                "model": model_name,
                "system_prompt": system_msg,
                "input": user_msg_text,
                "stream": True,
            }

            response_text = ""
            prompt_tokens = 0
            completion_tokens = 0
            response_id = "unknown"

            s = requests.Session()
            with s.post(api_url, json=data, stream=True) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue

                    line_str = line.decode("utf-8")
                    if line_str.startswith("data: "):
                        data_str = line_str[6:]
                        try:
                            event_data = json.loads(data_str)
                            event_type = event_data.get("type")

                            if event_type == "message.delta":
                                content = event_data.get("content", "")
                                response_text += content
                                yield content
                            elif event_type == "chat.end":
                                result_data = event_data.get("result", {})
                                stats = result_data.get("stats", {})
                                prompt_tokens = stats.get("input_tokens", 0)
                                completion_tokens = stats.get("total_output_tokens", 0)
                                response_id = result_data.get("response_id", "unknown")
                        except json.JSONDecodeError:
                            continue

            response_data = {
                "message": {"content": response_text},
                "id": f"lms-{response_id}",
                "model": model_name,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }

            if cache_key:
                self._save_to_cache(cache_key, model_name, response_text, response_data)
        else:
            litellm_messages = self._prepare_litellm_messages(messages_history)

            response = completion(
                model=model_name,
                messages=litellm_messages,
                stream=True,
            )

            chunks = []
            for chunk in response:
                chunks.append(chunk)
                content = chunk.choices[0].delta.content or ""
                if content:
                    yield content

            reconstructed_response = litellm.stream_chunk_builder(
                chunks, messages=litellm_messages
            )

            message_dict = reconstructed_response.choices[0].message.to_dict()
            response_text = reconstructed_response.choices[0].message.content
            message_dict.pop("content")
            message_dict.pop("role")

            msg_id = reconstructed_response.id
            model = reconstructed_response.model
            usage = reconstructed_response.usage.to_dict()

            response_data = {
                "message": message_dict,
                "id": msg_id,
                "model": model,
                "usage": usage,
            }

            if cache_key:
                self._save_to_cache(cache_key, model_name, response_text, response_data)

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
