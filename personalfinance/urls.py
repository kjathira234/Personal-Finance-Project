
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

    # Financial Analysis
     path(
        'analysis/',
        views.financial_analysis,
        name='financial_analysis'
    ),

path(
    'financial-details/',
    views.financial_details,
    name='financial_details'
),

path("edit-income/<int:id>/", views.edit_income, name="edit_income"),
path("delete-income/<int:id>/", views.delete_income, name="delete_income"),

path("edit-expense/<int:id>/", views.edit_expense, name="edit_expense"),
path("delete-expense/<int:id>/", views.delete_expense, name="delete_expense"),

path("edit-loan/<int:id>/",
     views.edit_loan,
     name="edit_loan"),

path("delete-loan/<int:id>/",
     views.delete_loan,
     name="delete_loan"),


# Dashboard
path(
    "dashboard/",
    views.dashboard,
    name="dashboard"
),
path("logout/", views.logout_view, name="logout"),
]

