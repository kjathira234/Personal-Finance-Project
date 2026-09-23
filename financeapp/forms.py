from django import forms
from .models import Income, Expense, Loan


class IncomeForm(forms.ModelForm):

    class Meta:
        model = Income
        fields = [
            'source',
            'amount',
            'date',
            'description'
        ]

        widgets = {
            'date': forms.DateInput(
                attrs={'type': 'date'}
            ),
        }


class ExpenseForm(forms.ModelForm):

    class Meta:
        model = Expense
        fields = [
            'category',
            'amount',
            'date',
            'description'
        ]

        widgets = {
            'date': forms.DateInput(
                attrs={'type': 'date'}
            ),
        }


class LoanForm(forms.ModelForm):

    class Meta:
        model = Loan
        fields = [
            'loan_type',
            'amount',
            'monthly_emi'
        ]