# :fire: django-llm-chat :fire:

This package is for calling LLMs. It supports all providers `litellm` supports.
It also supports SHA-256 based caching of LLM calls to save costs and reduce latency.

The goal is to have persistent `Chat`s with user and LLM messages plus some metadata like token usage that `litellm` returns for each LLM call.

## Installation

TODO

## Usage

```python
from django.contrib.auth import get_user_model

User = get_user_model()
user = User.objects.create_user(username="testuser", password="12345")
```

> [!TIP]
> Each  message has a foreign key to a user.
> A `Chat` can therefore have multiple users.

> [!NOTE]
> LLM messages have a user called `litellm`.

Now we create a `Chat`:

```python
from django_llm_chat.chat import Chat
chat = Chat.create()
```

This `Chat` will be linked to all the messages in this conversation. By default, for each new user message the whole chat history gets sent to the LLM.

You can add only one system message to a chat:

```python
system_msg = chat.create_system_message("You are an artist!", user)
```

You can create one or more user messages and call the LLM:

```python
from django_llm_chat.models import Message, LLMCall

# explicitly create a user message
first_user_msg = chat.create_user_message(text='Hi, how are you?', user=alex)

# below I create another one and send everything to the LLM

user_query = "Tell me who you are."
model_name = "ollama_chat/qwen3:4b"

ai_msg: Message
user_msg: Message
llm_call: LLMCall

# NOTE: user message gets created implicitly and all messages in chat history are sent to LLM
ai_msg, second_user_msg, llm_call = chat.call_llm(
    model_name=model_name,
    text=user_query,
    user=user,
    include_chat_history=True,
    use_cache=True,
    temperature=0.2,
    max_tokens=512,
)
print(ai_msg.text)  # prints LLM response text
```

You can also stream responses:

```python
gen = chat.stream_call_llm(
    model_name=model_name,
    text=user_query,
    user=user,
    include_chat_history=True,
    use_cache=True,
    temperature=0.2,
    max_tokens=512,
)

try:
    for chunk in gen:
        print(chunk, end="", flush=True)
except StopIteration as e:
    # After the stream ends, the generator returns ORM instances
    ai_msg, user_msg, llm_call = e.value
    print(f"\nFinal message saved: {ai_msg.id}")
```

> [!NOTE]
> `include_chat_history`, `temperature`, and `max_tokens` are optional parameters. Caching is not enabled by default.

## DSPy Usage

DSPy integration is available as an optional extra:

```bash
pip install "django-llm-chat[dspy]"
```

Use `DSPyChat` when you want DSPy calls to persist through the existing `Chat`, `Message`, and `LLMCall` tables without changing the current `Chat` API:

```python
import dspy

from django_llm_chat.dspy_chat import DSPyChat

dspy_chat = DSPyChat.create(project=None)
lm = dspy_chat.as_lm(
    model="openai/gpt-4o-mini",
    user=user,
    use_cache=True,
    temperature=0.2,
)

dspy.configure(lm=lm)
```

Each DSPy request message is stored as a `Message` in the bound chat, the request `LLMCall` is linked to those same `Message` rows, and the final assistant output is persisted as another `Message` linked to the same `LLMCall`.

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

# Note: remember `llm_call` was returned by `chat.call_llm`
llm_call = LLMCall.objects.get(llm_call.id)
print(llm_call.messages.all())
```

# Front-end Usage

This package includes a Django-based front-end for inspecting LLM calls and messages.

## Setup

Add `'django_llm_chat'` to your `INSTALLED_APPS`:

```python
INSTALLED_APPS = [
    # ... other apps
    'django_llm_chat',
]
```

Include the app URLs in your main `urls.py`:

```python
from django.urls import path, include

urlpatterns = [
    # ... other paths
    path('llm-chat/', include('django_llm_chat.urls')),
]
```

Visit `/llm-chat/` in your browser.

## Message Types

- **User Messages** (Blue): Messages sent by users
- **Assistant Messages** (Green): LLM responses  
- **System Messages** (Gray): System prompts and instructions

# Roadmap

Features and tasks I'm working on will be tracked using [this kanban board](https://github.com/sadeh-congnition/kanban).
