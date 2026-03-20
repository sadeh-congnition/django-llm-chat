import json
from unittest.mock import patch, MagicMock
from django.test import TestCase, override_settings
from django.contrib.auth import get_user_model
from django_llm_chat.chat import Chat
from django_llm_chat.models import Message, LLMCache

User = get_user_model()


class ChatTestCase(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="testuser", password="password")
        self.chat = Chat.create()

    @patch("django_llm_chat.backends.completion")
    def test_call_llm(self, mock_completion):
        # Mocking LiteLLM response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello, I am an AI"
        mock_response.choices[0].message.to_dict.return_value = {
            "role": "assistant",
            "content": "Hello, I am an AI",
        }
        mock_response.id = "test-id"
        mock_response.model = "gpt-3.5-turbo"
        mock_usage_dict = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }
        mock_response.usage.to_dict.return_value = mock_usage_dict
        mock_completion.return_value = mock_response

        # Call the method
        self.chat.call_llm(
            model_name="gpt-3.5-turbo",
            message="Hello AI",
            user=self.user,
            use_cache=True,
            include_chat_history=False,
        )

        # Assertions
        self.assertEqual(self.chat.last_llm_message.text, "Hello, I am an AI")
        self.assertEqual(self.chat.last_user_message.text, "Hello AI")
        self.assertIsNotNone(self.chat.llm_call)
        self.assertEqual(
            Message.objects.filter(chat=self.chat.chat_db_model).count(), 2
        )

        # Verify caching
        mock_completion.assert_called_once()
        self.assertTrue(LLMCache.objects.exists())

        # Call again to verify cache hit
        # We use a new chat object but the same text and no history to ensure same cache key
        new_chat = Chat.create()
        new_chat.call_llm(
            model_name="gpt-3.5-turbo",
            message="Hello AI",
            user=self.user,
            use_cache=True,
            include_chat_history=False,
        )
        self.assertEqual(new_chat.last_llm_message.text, "Hello, I am an AI")
        # Ensure completion was not called again
        mock_completion.assert_called_once()

    @patch("django_llm_chat.backends.completion")
    def test_stream_call_llm(self, mock_completion):
        # Mocking LiteLLM streaming response
        mock_chunk1 = MagicMock()
        mock_chunk1.choices = [MagicMock()]
        mock_chunk1.choices[0].delta.content = "Hello"

        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [MagicMock()]
        mock_chunk2.choices[0].delta.content = " World"

        mock_completion.return_value = [mock_chunk1, mock_chunk2]

        # Mocking stream_chunk_builder
        mock_reconstructed = MagicMock()
        mock_reconstructed.choices = [MagicMock()]
        mock_reconstructed.choices[0].message.content = "Hello World"
        mock_reconstructed.choices[0].message.to_dict.return_value = {
            "role": "assistant",
            "content": "Hello World",
        }
        mock_reconstructed.id = "stream-id"
        mock_reconstructed.model = "gpt-3.5-turbo"
        mock_usage_dict = {
            "prompt_tokens": 5,
            "completion_tokens": 5,
            "total_tokens": 10,
        }
        mock_reconstructed.usage.to_dict.return_value = mock_usage_dict

        with patch(
            "django_llm_chat.backends.litellm.stream_chunk_builder",
            return_value=mock_reconstructed,
        ):
            tokens = list(
                self.chat.stream_call_llm(
                    model_name="gpt-3.5-turbo", message="Stream this", user=self.user
                )
            )

            self.assertEqual(tokens, ["Hello", " World"])
            self.assertEqual(self.chat.last_llm_message.text, "Hello World")
            self.assertEqual(self.chat.last_user_message.text, "Stream this")
            self.assertEqual(
                Message.objects.filter(
                    chat=self.chat.chat_db_model, type=Message.Type.ASSISTANT
                ).count(),
                1,
            )
            msg = Message.objects.get(
                chat=self.chat.chat_db_model, type=Message.Type.ASSISTANT
            )
            self.assertEqual(msg.text, "Hello World")

    @patch("django_llm_chat.backends.requests.post")
    def test_call_llm_lmstudio(self, mock_post):
        # Mocking LM Studio response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "output": [{"type": "message", "content": "Hello from LM Studio"}],
            "stats": {"input_tokens": 5, "total_output_tokens": 10},
            "response_id": "test-lms-id",
        }
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        # Call the method with lm_studio prefix
        self.chat.call_llm(
            model_name="lm_studio/some-model",
            message="Hello LM",
            user=self.user,
        )

        # Assertions
        self.assertEqual(self.chat.last_llm_message.text, "Hello from LM Studio")
        mock_post.assert_called_once()

        # Verify provider used
        from django_llm_chat.backends import LMStudioProvider

        provider = self.chat._get_backend_provider("lm_studio/some-model")
        self.assertIsInstance(provider, LMStudioProvider)
