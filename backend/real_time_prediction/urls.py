from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

# Create a router and register the RealTimePredictionViewSet with it
router = DefaultRouter()
router.register('predictions', views.RealTimePredictionViewSet, basename='realtimeprediction')

urlpatterns = [
    # path('predict', views.fetchPrediction, name='predict'),
    # path('scrap', views.predictRealTime, name='predict'),
    path('', include(router.urls)),  # Include the DRF router URLs
]
