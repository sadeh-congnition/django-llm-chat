import pytest
import django
from django.conf import settings

if not settings.configured:
    settings.configure(
        DEBUG=True,
        DATABASES={
            "default": {
                "ENGINE": "django.db.backends.sqlite3",
                "NAME": ":memory:",
            }
        },
        INSTALLED_APPS=[
            "django.contrib.auth",
            "django.contrib.contenttypes",
            "django_llm_chat",
        ],
        SECRET_KEY="test-secret-key-for-streaming-tests",
        USE_TZ=True,
    )
    django.setup()
