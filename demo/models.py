
from django.db import models
from django.contrib.auth.models import User

class Subscription(models.Model):
    PLAN_CHOICES = [
        ('Free', 'Free'),
        ('Pro', 'Pro (£5)'),
        ('Enterprise', 'Enterprise (£10)'),
    ]
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    plan = models.CharField(max_length=32, choices=PLAN_CHOICES, default='Free')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} -> {self.plan} @ {self.created_at:%Y-%m-%d %H:%M}"

class Feedback(models.Model):
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    email = models.EmailField(blank=True)
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        who = self.user.username if self.user else self.email or 'anon'
        return f"Feedback by {who} @ {self.created_at:%Y-%m-%d %H:%M}"
