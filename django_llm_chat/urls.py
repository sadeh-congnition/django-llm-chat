from django.urls import path
from . import views

app_name = 'django_llm_chat'

urlpatterns = [
    path('', views.llm_call_list, name='llm_call_list'),
    path('call/<int:call_id>/', views.llm_call_detail, name='llm_call_detail'),
]
