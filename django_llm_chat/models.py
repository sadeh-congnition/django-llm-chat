from typing import Iterable, Self
from django.db import models
from django.conf import settings
from django.contrib.auth import get_user_model


class Project(models.Model):
    name = models.CharField(max_length=255)
    description = models.TextField(null=True, blank=True)
    date_created = models.DateTimeField(auto_now_add=True)
    date_updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name


class Chat(models.Model):
    project = models.ForeignKey(
        Project, on_delete=models.SET_NULL, null=True, blank=True, related_name="chats"
    )
    date_created = models.DateTimeField(auto_now_add=True)
    date_updated = models.DateTimeField(auto_now=True)
    input_tokens_count = models.IntegerField(default=0)
    output_tokens_count = models.IntegerField(default=0)

    @classmethod
    def get_service_user(cls, username: str):
        return get_user_model().objects.get(username=username)

    @classmethod
    def create_service_user(cls, username: str):
        return get_user_model().objects.create_user(
            username=username, password=username
        )

    @classmethod
    def get_or_create_service_user(cls, username: str):
        try:
            return cls.get_service_user(username)
        except get_user_model().DoesNotExist:
            return cls.create_service_user(username)

    @classmethod
    def get_llm_user(cls):
        return cls.get_service_user("llm")

    @classmethod
    def create_llm_user(cls):
        return cls.create_service_user("llm")

    def add_token_counts(self, input_token_count: int, output_token_count: int):
        self.input_tokens_count += input_token_count
        self.output_tokens_count += output_token_count
        self.save()

    def get_messages(self) -> models.QuerySet:
        return self.messages.order_by("date_created")


class Message(models.Model):
    class Type(models.TextChoices):
        USER = "user"
        ASSISTANT = "assistant"
        SYSTEM = "system"
        DEVELOPER = "developer"
        TOOL = "tool"

    type = models.CharField(max_length=10, choices=Type.choices)
    chat = models.ForeignKey(Chat, on_delete=models.CASCADE, related_name="messages")
    text = models.TextField()
    metadata = models.JSONField(default=dict, blank=True)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    date_created = models.DateTimeField(auto_now_add=True)
    date_updated = models.DateTimeField(auto_now=True)

    @classmethod
    def _create_message(
        cls,
        *,
        chat,
        text: str,
        user,
        message_type: str,
        metadata: dict | None = None,
    ) -> Self:
        return Message.objects.create(
            chat=chat,
            text=text,
            type=message_type,
            metadata=metadata or {},
            user=user,
        )

    @classmethod
    def create_user_message(
        cls, chat, text: str, user, metadata: dict | None = None
    ) -> Self:
        return cls._create_message(
            chat=chat,
            text=text,
            user=user,
            message_type=Message.Type.USER,
            metadata=metadata,
        )

    @classmethod
    def create_llm_message(
        cls, chat, text: str, user, metadata: dict | None = None
    ) -> Self:
        return cls._create_message(
            chat=chat,
            text=text,
            user=user,
            message_type=Message.Type.ASSISTANT,
            metadata=metadata,
        )

    @classmethod
    def create_system_message(
        cls, chat, text: str, user, metadata: dict | None = None
    ) -> Self:
        return cls._create_message(
            chat=chat,
            text=text,
            user=user,
            message_type=Message.Type.SYSTEM,
            metadata=metadata,
        )

    @classmethod
    def create_developer_message(
        cls, chat, text: str, user, metadata: dict | None = None
    ) -> Self:
        return cls._create_message(
            chat=chat,
            text=text,
            user=user,
            message_type=Message.Type.DEVELOPER,
            metadata=metadata,
        )

    @classmethod
    def create_tool_message(
        cls, chat, text: str, user, metadata: dict | None = None
    ) -> Self:
        return cls._create_message(
            chat=chat,
            text=text,
            user=user,
            message_type=Message.Type.TOOL,
            metadata=metadata,
        )


class LLMCall(models.Model):
    class Status(models.TextChoices):
        NEW = "new"
        GENERATION_IN_PROGRESS = "generation_in_progress"
        GENERATION_COMPLETED = "generation_completed"
        GENERATION_FAILED = "generation_failed"

    messages = models.ManyToManyField(Message)
    input_tokens_count = models.IntegerField(default=0)
    output_tokens_count = models.IntegerField(default=0)
    status = models.CharField(max_length=30, choices=Status.choices)
    request_data = models.JSONField(null=True, blank=True)
    response_data = models.JSONField(null=True, blank=True)
    error_data = models.JSONField(null=True, blank=True)

    @classmethod
    def create(
        cls, *messages: Iterable[Message], request_data: dict | None = None
    ) -> Self:
        db_model = LLMCall.objects.create(
            status=LLMCall.Status.NEW,
            request_data=request_data,
        )

        for m in messages:  # TODO optimize this to save all messages in one query
            db_model.add_message(m)

        return db_model

    def add_response_data(
        self, response_data: dict, input_token_count: int, output_token_count: int
    ):
        self.input_tokens_count += input_token_count
        self.output_tokens_count += output_token_count
        self.response_data = response_data
        self.status = self.Status.GENERATION_COMPLETED
        self.save()

    def mark_in_progress(self):
        self.status = self.Status.GENERATION_IN_PROGRESS
        self.save(update_fields=["status"])

    def mark_failed(self, error_data: dict):
        self.error_data = error_data
        self.status = self.Status.GENERATION_FAILED
        self.save(update_fields=["error_data", "status"])

    def add_message(self, message: Message):
        self.messages.add(message)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "response_data": self.response_data,
        }


class LLMCache(models.Model):
    cache_key = models.CharField(max_length=64, unique=True, db_index=True)
    model_name = models.CharField(max_length=255)
    response_text = models.TextField()
    response_data = models.JSONField()
    hit_count = models.IntegerField(default=0)
    date_created = models.DateTimeField(auto_now_add=True)
    date_updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.model_name} - {self.cache_key[:8]}"
