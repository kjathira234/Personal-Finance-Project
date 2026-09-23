
from django.contrib import admin
from django.urls import path
from financeapp import views


urlpatterns = [

    # Admin
    path(
        'admin/',
        admin.site.urls
    ),


    # Home
    path(
        '',
        views.home,
        name='home'
    ),

    path(
        'home/',
        views.home,
        name='home'
    ),


    # Login
    path(
        'login/',
        views.login_view,
        name='login'
    ),


    # Registration
    path(
        'registration/',
        views.registration,
        name='registration'
    ),


    # User Profile
    path(
        'profile/',
        views.user_profile,
        name='user_profile'
    ),


    # Income Tracker
    path(
        'income/',
        views.income_tracker,
        name='income_tracker'
    ),


    # Expense Tracker
    path(
        'expenses/',
        views.expense_tracker,
        name='expense_tracker'
    ),


    # Loan Information
    path(
        'loan/',
        views.loan_information,
        name='loan_information'
    ),


    # Financial Analysis
     path(
        'analysis/',
        views.financial_analysis,
        name='financial_analysis'
    ),


    # SVM Financial Health Report
    path(
        'financial-report/',
        views.financial_report,
        name='financial_report'
    ),


    # End-of-Month Report
    path(
        'monthly-report/',
        views.monthly_report,
        name='monthly_report'
    ),
    path(
    'loan-information/',
    views.loan_information,
    name='loan_information'
),
]

