
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('speech-to-sign/', views.speech_to_sign, name='speech_to_sign'),

    path('signup/', views.signup_view, name='signup'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),

    path('subscription/', views.subscription_view, name='subscription'),
    path('contact/', views.contact_view, name='contact'),
]
