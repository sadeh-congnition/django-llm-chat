# :fire: django-llm-chat :fire:

This package is for calling LLMs. It supports all providers `litellm` supports.

The goal is to have persistent `Chat`s with user and LLM messages plus some metadata like token usage that `litellm` returns for each LLM call.

# Installation

TODO

# Usage

```python
from django.contrib.auth import get_user_model

User = get_user_model()
user = User.objects.create_user(username="testuser", password="12345")
```

> [!TIP]
> Each user message has a foreign key to a user.
> A `Chat` can therefore have multiple users.

> [!NOTE]
> If you don't provide a user for user messages, then a default user will be created and assigned to the message.

> [!NOTE]
> LLM messages are linked to a default LLM user which gets created by default.

Now we create a `Chat`:

```python
from django_llm_chat.chat import Chat
chat = Chat.create()
```

This `Chat` will be linked to all the messages in this conversation. By default, for each new user message the whole chat history gets sent to the LLM.

```python
from django_llm_chat.models import Message, LLMCall

user_query = "Hi, tell me who you are."
model_name = "ollama_chat/qwen3:4b"

ai_msg: Message
user_msg: Message
llm_call: LLMCall
ai_msg, user_msg = chat.send_user_msg_to_llm(
    model_name=model_name, text=user_query, user=user, include_chat_history=True
)  # sends all messages in chat history to LLM

print(ai_msg.text)  # prints LLM response text
```

`user_msg` and `ai_message` are Django ORM model instances:

```python
class Message(models.Model):
    class Type(models.TextChoices):
        USER = "user"
        ASSISTANT = "assistant"
        SYSTEM = "system"

    type = models.CharField(max_length=10, choices=Type.choices)
    chat = models.ForeignKey(Chat, on_delete=models.CASCADE, related_name="messages")
    text = models.TextField()
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    date_created = models.DateTimeField(auto_now_add=True)
    date_updated = models.DateTimeField(auto_now=True)
```

The `Chat` ORM model keeps track of token usage:

```python
from django_llm_chat.model import Chat as ChatDBModel

chat = ChatDBModel.objects.get(id=chat.chat_db_model.id)
print(chat_db_model.input_tokens_count)  # total input tokens in this chat
print(chat_db_model.output_tokens_count)  # total output tokens in this chat
```

Now, let's say you see the LLM response and wonder what actually was sent to the LLM:

```python
from django_llm_chat.model import LLMCall

llm_call = LLMCall.objects.get(llm_call.id)
print(llm_call.messages.all())

```
