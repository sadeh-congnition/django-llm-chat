from django.contrib.auth import get_user_model
from django.test import RequestFactory, TestCase, override_settings

from django_llm_chat.models import Chat, LLMCall, Message
from django_llm_chat.views import llm_call_detail


User = get_user_model()

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {"context_processors": []},
    }
]


@override_settings(TEMPLATES=TEMPLATES, ROOT_URLCONF="django_llm_chat.tests.urls")
class LLMCallDetailViewTestCase(TestCase):
    def setUp(self):
        self.factory = RequestFactory()
        self.user = User.objects.create_user(username="testuser", password="password")
        self.chat = Chat.objects.create()

    def test_detail_renders_strict_bootstrap_accordion_markup(self):
        first_message = Message.create_user_message(
            self.chat,
            "First message with enough words to render a visible preview snippet.",
            self.user,
        )
        second_message = Message.create_llm_message(
            self.chat,
            "Second message with enough words to render a separate preview snippet.",
            self.user,
        )
        llm_call = LLMCall.create(first_message, second_message)

        response = llm_call_detail(
            self.factory.get(f"/call/{llm_call.id}/"), llm_call.id
        )

        self.assertEqual(response.status_code, 200)

        content = response.content.decode()
        self.assertIn('id="messagesAccordion"', content)
        self.assertEqual(content.count('data-bs-parent="#messagesAccordion"'), 2)

        for message in (first_message, second_message):
            self.assertIn(f'id="heading-{message.id}"', content)
            self.assertIn(f'data-bs-target="#collapse-{message.id}"', content)
            self.assertIn(f'aria-controls="collapse-{message.id}"', content)
            self.assertIn(f'id="collapse-{message.id}"', content)
            self.assertIn(f'aria-labelledby="heading-{message.id}"', content)
