from django.urls import path
from .views import DashboardView, PredictView, TrainModelView

app_name = 'analysis'

urlpatterns = [
    path('', DashboardView.as_view(), name='dashboard'),
    path('predict/', PredictView.as_view(), name='predict'),
    path('train/', TrainModelView.as_view(), name='train_model'),
]