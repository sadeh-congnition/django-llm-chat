from django.test import TestCase
from django.contrib.auth import get_user_model
from django_llm_chat.models import Project
from django_llm_chat.chat import Chat

User = get_user_model()


class ProjectTestCase(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="testuser", password="password")

    def test_project_creation(self):
        project = Project.objects.create(
            name="Test Project", description="A test project"
        )
        self.assertEqual(project.name, "Test Project")
        self.assertEqual(project.description, "A test project")
        self.assertEqual(str(project), "Test Project")

    def test_chat_with_project(self):
        project = Project.objects.create(name="Test Project")
        chat_wrapper = Chat.create(project=project)
        chat_db = chat_wrapper.chat_db_model

        self.assertEqual(chat_db.project, project)
        self.assertIn(chat_db, project.chats.all())

    def test_chat_without_project(self):
        chat_wrapper = Chat.create()
        chat_db = chat_wrapper.chat_db_model

        # Should be None by default
        self.assertIsNone(chat_db.project)
