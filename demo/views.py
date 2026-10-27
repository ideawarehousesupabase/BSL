
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.views.decorators.http import require_http_methods

from .models import Subscription, Feedback

def index(request):
    return render(request, 'demo/index.html', {'username': request.user.username if request.user.is_authenticated else ''})

def speech_to_sign(request):
    return render(request, 'demo/speech.html', {'username': request.user.username if request.user.is_authenticated else ''})

@require_http_methods(["GET", "POST"])
def signup_view(request):
    if request.method == 'POST':
        u = request.POST.get('username','').strip()
        p = request.POST.get('password','').strip()
        if not u or not p:
            messages.error(request, "Username and password required.")
        else:
            if User.objects.filter(username=u).exists():
                messages.error(request, "Username already exists.")
            else:
                User.objects.create_user(username=u, password=p)
                messages.success(request, "Account created.")
                return redirect('login')
    return render(request, 'demo/signup.html')

@require_http_methods(["GET", "POST"])
def login_view(request):
    if request.method == 'POST':
        u = request.POST.get('username','').strip()
        p = request.POST.get('password','').strip()
        user = authenticate(request, username=u, password=p)
        if user:
            login(request, user)
            messages.success(request, "Welcome back.")
            return redirect('index')
        else:
            messages.error(request, "Invalid credentials.")
    return render(request, 'demo/login.html')

def logout_view(request):
    logout(request)
    messages.info(request, "Signed out.")
    return redirect('index')

@login_required
@require_http_methods(["GET", "POST"])
def subscription_view(request):
    if request.method == 'POST':
        plan = request.POST.get('plan', 'Free')
        Subscription.objects.create(user=request.user, plan=plan)
        messages.success(request, f"Plan updated to {plan}.")
        return redirect('subscription')
    plans = Subscription.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'demo/subscription.html', {'plans': plans})

@require_http_methods(["GET", "POST"])
def contact_view(request):
    if request.method == 'POST':
        email = request.POST.get('email', '')
        message = request.POST.get('message', '')
        Feedback.objects.create(user=request.user if request.user.is_authenticated else None, email=email, message=message)
        messages.success(request, "Thanks! We'll get back to you.")
        return redirect('contact')
    return render(request, 'demo/contact.html')
