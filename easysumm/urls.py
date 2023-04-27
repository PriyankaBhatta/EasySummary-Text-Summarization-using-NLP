from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path
from easysumm import views

urlpatterns = [
    path('', views.home, name="home"),  
    path('summarizenow/', views.summarizenow, name='summarizenow'),
    path('sentiment/', views.sentiment, name='sentiment'),
    path('ner/', views.ner, name='ner')
    
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
