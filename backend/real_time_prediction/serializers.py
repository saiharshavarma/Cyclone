# serializers.py
from rest_framework import serializers
from .models import RealTimePrediction

class RealTimePredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = RealTimePrediction
        fields = ['id', 'wind', 'pressure', 't_number', 'category', 'original_img', 'processed_img', 'timestamp']
