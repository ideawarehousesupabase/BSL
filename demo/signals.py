
from django.conf import settings
from django.contrib.auth.models import User
from django.db.models.signals import post_migrate
from django.dispatch import receiver

@receiver(post_migrate)
def create_default_admin(sender, **kwargs):
    if sender.name != 'demo':
        return
    username = getattr(settings, "AUTO_ADMIN_USERNAME", "admin")
    email = getattr(settings, "AUTO_ADMIN_EMAIL", "admin@example.com")
    password = getattr(settings, "AUTO_ADMIN_PASSWORD", "admin123")
    if not User.objects.filter(username=username).exists():
        try:
            User.objects.create_superuser(username=username, email=email, password=password)
            print(f"[demo] Created default admin: {username}/{password}")
        except Exception as e:
            print("[demo] Admin create error:", e)
