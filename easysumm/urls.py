from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path
from easysumm import views

urlpatterns = [
    path('', views.home, name="home"),  
    path('summarizenow/', views.summarizenow, name='summarizenow'),
    
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
