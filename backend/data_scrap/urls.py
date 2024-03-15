from django.urls import path
from . import views

urlpatterns = [
    path('', views.fetchRealTimeData, name='data_scrap'),
]
