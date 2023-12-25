from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    
   
    
    path('sentiment/',views.sentiment,name="sentiment"),
    path('sentimentDL/',views.sentimentDL,name="sentimentDL"),
    path('bi/',views.SentimentBi,name="bi"),


## partie investor (mehdi)
    path('investorPrediction/',views.result_investor_prediction, name='investorPrediction'),
    path('InvestorDashboard/',views.InvestorBi,name="InvestorDashboard"),
    path('tradePrediction/', views.result, name='investor'),
    
]
