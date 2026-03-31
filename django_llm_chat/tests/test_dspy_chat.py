import asyncio
import copy
import importlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TransactionTestCase

from django_llm_chat.models import LLMCache, LLMCall, Message

User = get_user_model()


def build_fake_dspy_module():
    fake_dspy = types.ModuleType("dspy")

    class FakeLM:
        def __init__(
            self,
            model,
            model_type="chat",
            temperature=0.0,
            max_tokens=1000,
            cache=True,
            use_developer_role=False,
            **kwargs,
        ):
            self.model = model
            self.model_type = model_type
            self.cache = cache
            self.use_developer_role = use_developer_role
            self.kwargs = dict(
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            self.history = []

        def forward(self, prompt=None, messages=None, **kwargs):
            raise NotImplementedError

        async def aforward(self, prompt=None, messages=None, **kwargs):
            raise NotImplementedError

        def copy(self, **kwargs):
            new_instance = copy.deepcopy(self)
            new_instance.history = []
            for key, value in kwargs.items():
                if hasattr(new_instance, key):
                    setattr(new_instance, key, value)
                else:
                    new_instance.kwargs[key] = value
            return new_instance

        def update_history(self, entry):
            self.history.append(entry)

    fake_dspy.LM = FakeLM
    return fake_dspy


def build_chat_response(
    *,
    content="Hello from DSPy",
    prompt_tokens=10,
    completion_tokens=6,
    tool_calls=None,
):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message)
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    return SimpleNamespace(
        id="chat-response-id",
        model="openai/gpt-4o-mini",
        choices=[choice],
        usage=usage,
        cache_hit=False,
        _hidden_params={},
    )


class DSPyChatTestCase(TransactionTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.fake_dspy = build_fake_dspy_module()
        cls.sys_modules_patch = patch.dict(sys.modules, {"dspy": cls.fake_dspy})
        cls.sys_modules_patch.start()
        import django_llm_chat.dspy_chat as dspy_chat_module

        cls.dspy_chat_module = importlib.reload(dspy_chat_module)

    @classmethod
    def tearDownClass(cls):
        cls.sys_modules_patch.stop()
        super().tearDownClass()

    def setUp(self):
        self.user = User.objects.create_user(username="testuser", password="password")

    def test_dspy_chat_persists_prompt_messages_and_response(self):
        dspy_chat = self.dspy_chat_module.DSPyChat.create()

        with patch.object(
            self.fake_dspy.LM,
            "forward",
            return_value=build_chat_response(),
        ) as mock_forward:
            lm = dspy_chat.as_lm(model="openai/gpt-4o-mini", user=self.user)
            response = lm.forward(
                messages=[
                    {"role": "system", "content": "You are terse."},
                    {"role": "user", "content": "Say hi"},
                ]
            )

        self.assertEqual(response.choices[0].message.content, "Hello from DSPy")
        self.assertEqual(mock_forward.call_count, 1)
        self.assertEqual(dspy_chat.chat_db_model.messages.count(), 3)

        llm_call = LLMCall.objects.get()
        persisted_messages = list(
            llm_call.messages.order_by("date_created", "id").values_list("type", "text")
        )
        self.assertEqual(
            persisted_messages,
            [
                (Message.Type.SYSTEM, "You are terse."),
                (Message.Type.USER, "Say hi"),
                (Message.Type.ASSISTANT, "Hello from DSPy"),
            ],
        )
        self.assertEqual(llm_call.status, LLMCall.Status.GENERATION_COMPLETED)
        self.assertEqual(llm_call.request_data["integration"], "dspy")
        self.assertEqual(dspy_chat.chat_db_model.input_tokens_count, 10)
        self.assertEqual(dspy_chat.chat_db_model.output_tokens_count, 6)

    def test_dspy_chat_lm_copy_preserves_bound_chat(self):
        dspy_chat = self.dspy_chat_module.DSPyChat.create()
        lm = dspy_chat.as_lm(
            model="openai/gpt-4o-mini", user=self.user, temperature=0.2
        )

        copied = lm.copy(temperature=0.7)

        self.assertIs(copied.chat, lm.chat)
        self.assertIs(copied.user, lm.user)
        self.assertEqual(copied.kwargs["temperature"], 0.7)

    def test_dspy_chat_uses_existing_llm_cache_model(self):
        dspy_chat = self.dspy_chat_module.DSPyChat.create()

        with patch.object(
            self.fake_dspy.LM,
            "forward",
            return_value=build_chat_response(content="Cached hello"),
        ) as mock_forward:
            lm = dspy_chat.as_lm(model="openai/gpt-4o-mini", user=self.user)
            lm.forward(messages=[{"role": "user", "content": "Repeat me"}])
            lm.forward(messages=[{"role": "user", "content": "Repeat me"}])

        self.assertEqual(mock_forward.call_count, 1)
        cache_item = LLMCache.objects.get()
        self.assertEqual(cache_item.hit_count, 1)
        self.assertEqual(LLMCall.objects.count(), 2)
        self.assertEqual(dspy_chat.chat_db_model.input_tokens_count, 10)
        self.assertEqual(dspy_chat.chat_db_model.output_tokens_count, 6)

    def test_dspy_chat_persists_tool_messages_and_metadata(self):
        dspy_chat = self.dspy_chat_module.DSPyChat.create()

        with patch.object(
            self.fake_dspy.LM,
            "forward",
            return_value=build_chat_response(content="Final answer"),
        ):
            lm = dspy_chat.as_lm(model="openai/gpt-4o-mini", user=self.user)
            lm.forward(
                messages=[
                    {"role": "user", "content": "Use the calculator"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"id": "call_1", "name": "calculator"}],
                    },
                    {
                        "role": "tool",
                        "content": {"result": 4},
                        "tool_call_id": "call_1",
                    },
                ]
            )

        persisted_messages = list(
            dspy_chat.chat_db_model.messages.order_by("date_created", "id")
        )
        self.assertEqual(
            [message.type for message in persisted_messages],
            [
                Message.Type.USER,
                Message.Type.ASSISTANT,
                Message.Type.TOOL,
                Message.Type.ASSISTANT,
            ],
        )
        self.assertEqual(
            persisted_messages[1].metadata["tool_calls"][0]["name"], "calculator"
        )
        self.assertEqual(persisted_messages[2].metadata["tool_call_id"], "call_1")
        self.assertEqual(persisted_messages[2].text, '{"result": 4}')

    def test_dspy_chat_async_path_persists_response(self):
        dspy_chat = self.dspy_chat_module.DSPyChat.create()

        async def fake_aforward(*args, **kwargs):
            return build_chat_response(content="Async hello")

        with patch.object(self.fake_dspy.LM, "aforward", side_effect=fake_aforward):
            lm = dspy_chat.as_lm(model="openai/gpt-4o-mini", user=self.user)
            response = asyncio.run(
                lm.aforward(messages=[{"role": "user", "content": "Async please"}])
            )

        self.assertEqual(response.choices[0].message.content, "Async hello")
        self.assertEqual(dspy_chat.chat_db_model.messages.count(), 2)
        self.assertEqual(LLMCall.objects.count(), 1)
