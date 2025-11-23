```python
from django.contrib.auth import get_user_model
from django_llm_chat.chat import Chat
from django_llm_chat.model import Chat as ChatDBModel, LLMCall

User = get_user_model()
user = User.objects.create_user(username="testuser", password="12345")

chat = Chat.create()

user_query = "Hi, tell me who you are."
ai_msg = chat.send_user_msg_to_llm(
    model_name=model_name, text=user_query, user=user, include_chat_history=True
)  # sends all messages in chat history to LLM

print(ai_msg.text)  # return LLM response

chat = ChatDBModel.objects.get(id=chat.chat_db_model.id)
print(chat_db_model.input_tokens_count)  # total input tokens in this chat
print(chat_db_model.output_tokens_count)  # total output tokens in this chat

llm_call = LLMCall.objects.first()
print(llm_call.to_dict())

```
