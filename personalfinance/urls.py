

from django.contrib import admin
from django.urls import path
from financeapp import views

urlpatterns = [
    path('admin/', admin.site.urls),

    path('prediction/',views.prediction,name='prediction'),

    #home page
    path("home", views.home, name="home"),

    #Login
    path("login/", views.login_view, name="login"),

    #Registration
    path("registration/", views.registration, name="registration"),

    # Financial assessment
    path("assessment/", views.assessment, name="assessment"),

    # Budget recommendation
    path("recommendation/", views.recommendation, name="recommendation"),

    # Risk management
    path("riskmanagement/", views.riskmanagement, name="riskmanagement"),

    # AI chatbot
    path("chatbot/", views.chatbot, name="chatbot"),

    # Dashboard
    path("dashboard/", views.dashboard, name="dashboard"),

]
