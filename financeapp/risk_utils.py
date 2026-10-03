from django.db.models import Sum
from .models import Expense


def calculate_expense_risk(user, total_income):

    expenses = Expense.objects.filter(
        user=user
    )

    category_totals = expenses.values(
        'category'
    ).annotate(
        total=Sum('amount')
    ).order_by('-total')

    risks = []

    if total_income <= 0:
        return risks

    for item in category_totals:

        category = item['category']
        amount = float(item['total'])

        percentage = (
            amount / float(total_income)
        ) * 100

        if percentage >= 20:

            risks.append({

                'category': category,

                'amount': amount,

                'percentage': round(
                    percentage, 2
                ),

                'message':
                f'{category} spending is high.'
            })

    return risks