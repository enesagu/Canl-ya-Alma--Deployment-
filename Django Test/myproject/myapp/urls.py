from django.urls import path
from .views import PredictView, PredictAPIView

urlpatterns = [
    path('predict/', PredictView.as_view(), name='predict-view'),
    path('api/predict/', PredictAPIView.as_view(), name='predict-api-view'),
]
