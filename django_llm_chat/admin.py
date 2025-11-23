from django.contrib import admin

from .models import Chat, Message, LLMCall


@admin.register(Chat)
class ChatAdmin(admin.ModelAdmin):
    list_display = ("id",)


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "chat", "type", "text")


@admin.register(LLMCall)
class LLMCallAdmin(admin.ModelAdmin):
    list_display = ("id", "status", "response_data")
