from django.urls import include, path


urlpatterns = [
    path(
        "",
        include(
            ("django_llm_chat.urls", "django_llm_chat"), namespace="django_llm_chat"
        ),
    ),
]
