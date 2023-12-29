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
    path('tradePrediction/', views.results, name='investor'),
    path('template/',views.my_view,name="template"),


    ## partie candle (sarra)
    ##path('CandlePrediction/',views.getPredictionss, name='CandlePrediction'),
    path('CandleDashboard/',views.CandleBi,name="CandleDashboard"),
    path('CandlePrediction/', views.result, name='CandlePrediction'),

    ## partie pepsi (sarra)
     path('PepsiDashboard/',views.PepsiBi,name="PepsiDashboard"),
]
