import hashlib
import json
from typing import Iterable
from pydantic import BaseModel
from .models import LLMCache, Message


class LLMCacheService:
    @staticmethod
    def _normalize_for_cache(value):
        if isinstance(value, type) and issubclass(value, BaseModel):
            return {
                "__pydantic_model__": value.__name__,
                "schema": value.model_json_schema(),
            }
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        if isinstance(value, Message):
            return {
                "role": value.type,
                "content": value.text,
                "metadata": value.metadata,
            }
        if isinstance(value, dict):
            return {
                str(key): LLMCacheService._normalize_for_cache(val)
                for key, val in sorted(value.items())
            }
        if isinstance(value, (list, tuple)):
            return [LLMCacheService._normalize_for_cache(item) for item in value]
        return value

    @staticmethod
    def compute_request_cache_key(request_data: dict) -> str:
        normalized = LLMCacheService._normalize_for_cache(request_data)
        key_str = json.dumps(normalized, sort_keys=True)
        return hashlib.sha256(key_str.encode("utf-8")).hexdigest()

    @staticmethod
    def compute_cache_key(
        model_name: str,
        messages: Iterable[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
        output_model: type[BaseModel] | None = None,
    ) -> str:
        msg_data = list(messages)
        return LLMCacheService.compute_request_cache_key(
            {
                "model": model_name,
                "messages": msg_data,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "output_model": (
                    output_model.model_json_schema()
                    if output_model is not None
                    else None
                ),
            }
        )

    @staticmethod
    def lookup_cache(cache_key: str) -> LLMCache | None:
        try:
            cache_item = LLMCache.objects.get(cache_key=cache_key)
            cache_item.hit_count += 1
            cache_item.save(update_fields=["hit_count"])
            return cache_item
        except LLMCache.DoesNotExist:
            return None

    @staticmethod
    def save_to_cache(
        cache_key: str, model_name: str, response_text: str, response_data: dict
    ):
        LLMCache.objects.get_or_create(
            cache_key=cache_key,
            defaults={
                "model_name": model_name,
                "response_text": response_text,
                "response_data": response_data,
            },
        )
