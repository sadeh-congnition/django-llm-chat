from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any, Iterable

from asgiref.sync import sync_to_async

from .models import Chat as ChatDBModel
from .models import LLMCall, Message, Project
from .services import LLMCacheService

try:
    import dspy
except ImportError:  # pragma: no cover - exercised in tests via import guards.
    dspy = None


class MissingDSPyDependencyError(ImportError):
    pass


class AttrDict(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover - defensive.
            raise AttributeError(item) from exc

    __setattr__ = dict.__setitem__


def _to_plain_data(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _to_plain_data(value.model_dump())
    if hasattr(value, "to_dict"):
        return _to_plain_data(value.to_dict())
    if isinstance(value, dict):
        return {key: _to_plain_data(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_data(item) for item in value]
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return {
            key: _to_plain_data(val)
            for key, val in vars(value).items()
            if not key.startswith("_")
        }
    return value


def _to_attr_dict(value: Any) -> Any:
    if isinstance(value, dict):
        converted = AttrDict()
        for key, val in value.items():
            converted[key] = _to_attr_dict(val)
        return converted
    if isinstance(value, list):
        return [_to_attr_dict(item) for item in value]
    return value


def _stringify_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(_to_plain_data(value), sort_keys=True)


def _usage_to_dict(response: Any) -> dict[str, int]:
    usage = _to_plain_data(getattr(response, "usage", {}) or {})
    prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
    completion_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))
    total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


@dataclass
class DSPyChat:
    chat_db_model: ChatDBModel
    llm_user: object
    dspy_user: object
    last_user_message: Message | None = None
    llm_call: LLMCall | None = None
    last_llm_message: Message | None = None

    @classmethod
    def create(cls, project: Project | None = None) -> "DSPyChat":
        db_model = ChatDBModel.objects.create(project=project)
        return cls.from_db(db_model)

    @classmethod
    def from_db(cls, chat_db_model: ChatDBModel) -> "DSPyChat":
        return cls(
            chat_db_model=chat_db_model,
            llm_user=ChatDBModel.get_or_create_service_user("llm"),
            dspy_user=ChatDBModel.get_or_create_service_user("dspy"),
        )

    def get_msg_history(self) -> Iterable[Message]:
        return self.chat_db_model.get_messages().all()

    def create_llm_call(
        self, *messages: Iterable[Message], request_data: dict | None = None
    ) -> LLMCall:
        return LLMCall.create(*messages, request_data=request_data)

    def add_tokens(self, input_token_count: int, output_token_count: int):
        self.chat_db_model.add_token_counts(input_token_count, output_token_count)

    def as_lm(
        self,
        model: str,
        user: object | None = None,
        use_cache: bool = True,
        **lm_kwargs,
    ):
        if dspy is None:
            raise MissingDSPyDependencyError(
                "Install the optional 'dspy' dependency to use DSPyChat."
            )
        return DSPyChatLM(
            chat=self,
            model=model,
            user=user,
            use_cache=use_cache,
            **lm_kwargs,
        )


if dspy is None:

    class DSPyChatLM:  # pragma: no cover - import guard only.
        def __init__(self, *args, **kwargs):
            raise MissingDSPyDependencyError(
                "Install the optional 'dspy' dependency to use DSPyChatLM."
            )

else:

    class DSPyChatLM(dspy.LM):
        def __init__(
            self,
            *,
            chat: DSPyChat,
            model: str,
            user: object | None = None,
            use_cache: bool = True,
            **kwargs,
        ):
            kwargs = dict(kwargs)
            kwargs.pop("cache", None)
            super().__init__(model=model, cache=False, **kwargs)
            self.chat = chat
            self.user = user
            self.use_cache = use_cache
            self._last_persisted_context: dict[str, Any] | None = None

        def copy(self, **kwargs):
            new_instance = super().copy(**kwargs)
            new_instance.chat = self.chat
            new_instance.user = self.user
            new_instance.use_cache = self.use_cache
            new_instance._last_persisted_context = None
            return new_instance

        def update_history(self, entry):
            if self._last_persisted_context:
                entry = dict(entry)
                entry["django_llm_chat"] = dict(self._last_persisted_context)
            super().update_history(entry)

        def forward(
            self,
            prompt: str | None = None,
            messages: list[dict[str, Any]] | None = None,
            **kwargs,
        ):
            request_messages = self._normalize_messages(prompt, messages)
            request_kwargs = self._sanitize_request_kwargs(kwargs)
            return self._execute_sync(prompt, request_messages, request_kwargs)

        async def aforward(
            self,
            prompt: str | None = None,
            messages: list[dict[str, Any]] | None = None,
            **kwargs,
        ):
            request_messages = self._normalize_messages(prompt, messages)
            request_kwargs = self._sanitize_request_kwargs(kwargs)
            return await self._execute_async(prompt, request_messages, request_kwargs)

        def _normalize_messages(
            self,
            prompt: str | None,
            messages: list[dict[str, Any]] | None,
        ) -> list[dict[str, Any]]:
            normalized = copy.deepcopy(
                messages or [{"role": "user", "content": prompt or ""}]
            )
            if self.use_developer_role and self.model_type == "responses":
                normalized = [
                    {**message, "role": "developer"}
                    if message.get("role") == "system"
                    else message
                    for message in normalized
                ]
            return normalized

        def _sanitize_request_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
            sanitized = dict(kwargs)
            sanitized.pop("cache", None)
            return sanitized

        def _build_request_data(
            self, messages: list[dict[str, Any]], kwargs: dict[str, Any]
        ) -> dict[str, Any]:
            return {
                "integration": "dspy",
                "model": self.model,
                "model_type": self.model_type,
                "messages": _to_plain_data(messages),
                "kwargs": _to_plain_data({**self.kwargs, **kwargs}),
            }

        def _get_prompt_user(self):
            return self.user or self.chat.dspy_user

        def _get_message_user(self, role: str):
            if role == Message.Type.ASSISTANT:
                return self.chat.llm_user
            if role == Message.Type.USER:
                return self._get_prompt_user()
            return self.chat.dspy_user

        def _create_message_from_request(self, message_data: dict[str, Any]) -> Message:
            role = message_data.get("role", Message.Type.USER)
            metadata = {
                key: _to_plain_data(value)
                for key, value in message_data.items()
                if key not in {"role", "content"}
            }
            content = message_data.get("content", "")
            if not isinstance(content, str):
                metadata["content"] = _to_plain_data(content)
            text = _stringify_content(content)
            user = self._get_message_user(role)

            if role == Message.Type.SYSTEM:
                return Message.create_system_message(
                    self.chat.chat_db_model, text, user, metadata=metadata
                )
            if role == Message.Type.ASSISTANT:
                return Message.create_llm_message(
                    self.chat.chat_db_model, text, user, metadata=metadata
                )
            if role == Message.Type.DEVELOPER:
                return Message.create_developer_message(
                    self.chat.chat_db_model, text, user, metadata=metadata
                )
            if role == Message.Type.TOOL:
                return Message.create_tool_message(
                    self.chat.chat_db_model, text, user, metadata=metadata
                )

            created = Message.create_user_message(
                self.chat.chat_db_model, text, user, metadata=metadata
            )
            self.chat.last_user_message = created
            return created

        def _persist_prompt_messages(
            self, request_messages: list[dict[str, Any]]
        ) -> list[Message]:
            return [
                self._create_message_from_request(message_data)
                for message_data in request_messages
            ]

        def _extract_completion_payload(
            self, response: Any
        ) -> tuple[str, dict[str, Any]]:
            plain_response = _to_plain_data(response)
            if "output" in plain_response:
                text_parts = []
                tool_calls = []
                for item in plain_response.get("output", []):
                    item_type = item.get("type")
                    if item_type == "message":
                        for content_item in item.get("content", []):
                            text_parts.append(content_item.get("text", ""))
                    elif item_type == "function_call":
                        tool_calls.append(item)
                metadata = {"response_format": "responses"}
                if tool_calls:
                    metadata["tool_calls"] = tool_calls
                return "".join(text_parts), metadata

            choice = (plain_response.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            metadata = {
                key: value
                for key, value in message.items()
                if key not in {"role", "content"}
            }
            return message.get("content", "") or "", metadata

        def _build_cached_response(self, response_data: dict[str, Any]):
            cached_payload = copy.deepcopy(response_data)
            usage = cached_payload.get("usage", {})
            if usage:
                usage["prompt_tokens"] = 0
                usage["completion_tokens"] = 0
                usage["total_tokens"] = 0
            cached_payload["cache_hit"] = True
            cached_payload.setdefault("_hidden_params", {})
            return _to_attr_dict(cached_payload)

        def _record_success(
            self,
            llm_call: LLMCall,
            response: Any,
            cache_key: str | None,
        ):
            response_data = _to_plain_data(response)
            usage = _usage_to_dict(response)
            input_tokens = usage["prompt_tokens"]
            output_tokens = usage["completion_tokens"]

            if cache_key and not getattr(response, "cache_hit", False):
                response_text, _ = self._extract_completion_payload(response)
                LLMCacheService.save_to_cache(
                    cache_key, self.model, response_text, response_data
                )

            self.chat.add_tokens(input_tokens, output_tokens)
            llm_call.add_response_data(response_data, input_tokens, output_tokens)

            response_text, response_metadata = self._extract_completion_payload(
                response
            )
            llm_message = Message.create_llm_message(
                self.chat.chat_db_model,
                response_text,
                self.chat.llm_user,
                metadata=response_metadata,
            )
            llm_call.add_message(llm_message)
            self.chat.llm_call = llm_call
            self.chat.last_llm_message = llm_message
            self._last_persisted_context = {
                "chat_id": self.chat.chat_db_model.id,
                "llm_call_id": llm_call.id,
                "message_ids": list(
                    llm_call.messages.order_by("date_created", "id").values_list(
                        "id", flat=True
                    )
                ),
            }

        def _record_failure(self, llm_call: LLMCall, exc: Exception):
            llm_call.mark_failed(
                {
                    "type": exc.__class__.__name__,
                    "message": str(exc),
                }
            )

        def _execute_sync(
            self,
            prompt: str | None,
            request_messages: list[dict[str, Any]],
            request_kwargs: dict[str, Any],
        ):
            prompt_messages = self._persist_prompt_messages(request_messages)
            request_data = self._build_request_data(request_messages, request_kwargs)
            llm_call = self.chat.create_llm_call(
                *prompt_messages, request_data=request_data
            )
            llm_call.mark_in_progress()

            cache_key = None
            try:
                response = None
                if self.use_cache:
                    cache_key = LLMCacheService.compute_request_cache_key(request_data)
                    cache_item = LLMCacheService.lookup_cache(cache_key)
                    if cache_item:
                        response = self._build_cached_response(cache_item.response_data)

                if response is None:
                    response = super().forward(
                        prompt=prompt,
                        messages=request_messages,
                        cache=False,
                        **request_kwargs,
                    )

                self._record_success(llm_call, response, cache_key)
                return response
            except Exception as exc:
                self._record_failure(llm_call, exc)
                raise

        async def _execute_async(
            self,
            prompt: str | None,
            request_messages: list[dict[str, Any]],
            request_kwargs: dict[str, Any],
        ):
            prompt_messages = await sync_to_async(
                self._persist_prompt_messages,
                thread_sensitive=True,
            )(request_messages)
            request_data = self._build_request_data(request_messages, request_kwargs)
            llm_call = await sync_to_async(
                self.chat.create_llm_call,
                thread_sensitive=True,
            )(*prompt_messages, request_data=request_data)
            await sync_to_async(llm_call.mark_in_progress, thread_sensitive=True)()

            cache_key = None
            try:
                response = None
                if self.use_cache:
                    cache_key = LLMCacheService.compute_request_cache_key(request_data)
                    cache_item = await sync_to_async(
                        LLMCacheService.lookup_cache,
                        thread_sensitive=True,
                    )(cache_key)
                    if cache_item:
                        response = self._build_cached_response(cache_item.response_data)

                if response is None:
                    response = await super().aforward(
                        prompt=prompt,
                        messages=request_messages,
                        cache=False,
                        **request_kwargs,
                    )

                await sync_to_async(
                    self._record_success,
                    thread_sensitive=True,
                )(llm_call, response, cache_key)
                return response
            except Exception as exc:
                await sync_to_async(
                    self._record_failure,
                    thread_sensitive=True,
                )(llm_call, exc)
                raise
