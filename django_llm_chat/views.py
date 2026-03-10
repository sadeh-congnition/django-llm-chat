from django.shortcuts import render, get_object_or_404
from django.db.models import Prefetch
from .models import LLMCall, Message


def llm_call_list(request):
    """View to display a list of all LLM calls."""
    llm_calls = LLMCall.objects.select_related().prefetch_related(
        Prefetch('messages', queryset=Message.objects.order_by('date_created'))
    ).order_by('-id')
    
    return render(request, 'django_llm_chat/llm_call_list.html', {
        'llm_calls': llm_calls
    })


def llm_call_detail(request, call_id):
    """View to display details of a specific LLM call including all messages."""
    llm_call = get_object_or_404(
        LLMCall.objects.select_related().prefetch_related(
            Prefetch('messages', queryset=Message.objects.order_by('date_created'))
        ),
        id=call_id
    )
    
    return render(request, 'django_llm_chat/llm_call_detail.html', {
        'llm_call': llm_call
    })
