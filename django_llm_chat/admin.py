from django.contrib import admin

from .models import Chat, Message, LLMCall, LLMCache


@admin.register(Chat)
class ChatAdmin(admin.ModelAdmin):
    list_display = ("id",)


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "chat", "type", "text")


@admin.register(LLMCall)
class LLMCallAdmin(admin.ModelAdmin):
    list_display = ("id", "status", "response_data")


@admin.register(LLMCache)
class LLMCacheAdmin(admin.ModelAdmin):
    list_display = ("cache_key", "model_name", "hit_count", "date_created")
    readonly_fields = ("cache_key", "date_created", "date_updated")
