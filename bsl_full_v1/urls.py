
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('demo.urls')),
]

# Serve the sign clips during development (use a web server or object storage in production)
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
