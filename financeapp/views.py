
import pandas as pd
import joblib
import os
import json

from datetime import datetime

from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.db.models import Sum
from django.contrib.auth import authenticate, login, logout
from django.conf import settings
from django.contrib.auth import logout

from .models import Income, Expense, Loan
from .forms import IncomeForm, ExpenseForm, LoanForm
from django.shortcuts import get_object_or_404, redirect, render
from django.db.models.functions import TruncMonth




BEST_MODEL_PATH = (
    settings.BASE_DIR
    / "models"
    / "best_financial_status_model.pkl"
)

try:
    best_model = joblib.load(BEST_MODEL_PATH)
    print("Logistic Regression model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)
    best_model = None


# ============================================================
# HOME
# ============================================================

def home(request):

    return render(
        request,
        "home.html"
    )
    
def logout_view(request):
    logout(request)
    return redirect("login")    


# ============================================================
# LOGIN
# ============================================================

def login_view(request):

    if request.method == "POST":

        # ====================================================
        # GET LOGIN DETAILS
        # ====================================================

        username = request.POST.get("username")

        password = request.POST.get("password")

        # ====================================================
        # AUTHENTICATE USER
        # ====================================================

        user = authenticate(
            request,
            username=username,
            password=password
        )

        # ====================================================
        # LOGIN SUCCESS
        # ====================================================

        if user is not None:

            # =================================================
            # GET PROFILE DATA FROM SESSION
            # =================================================

            age = request.session.get("age")

            gender = request.session.get("gender")

            education_level = request.session.get(
                "education_level"
            )

            employment_status = request.session.get(
                "employment_status"
            )

            # =================================================
            # LOGIN USER
            # =================================================

            login(
                request,
                user
            )

            # =================================================
            # RESTORE PROFILE DATA
            # =================================================

            if age is not None:

                request.session["age"] = age

            if gender is not None:

                request.session["gender"] = gender

            if education_level is not None:

                request.session[
                    "education_level"
                ] = education_level

            if employment_status is not None:

                request.session[
                    "employment_status"
                ] = employment_status

            request.session.modified = True

            # =================================================
            # GO TO FINANCIAL DETAILS
            # =================================================

            return redirect(
                "financial_details"
            )

        # ====================================================
        # LOGIN FAILED
        # ====================================================

        return render(
            request,
            "login.html",
            {
                "error":
                    "Invalid username or password"
            }
        )

    # ========================================================
    # DISPLAY LOGIN PAGE
    # ========================================================

    return render(
        request,
        "login.html"
    )


# ============================================================
# EXPENSE TRACKER
# ============================================================

def expense_tracker(request):

    if not request.user.is_authenticated:

        return redirect(
            "login"
        )


    if request.method == "POST":

        form = ExpenseForm(
            request.POST
        )

        if form.is_valid():

            expense = form.save(
                commit=False
            )

            expense.user = request.user

            expense.save()

            return redirect(
                "expense_tracker"
            )

    else:

        form = ExpenseForm()


    expenses = (
        Expense.objects
        .filter(
            user=request.user
        )
        .order_by(
            "-date",
            "-id"
        )
    )


    return render(
        request,
        "expense_tracker.html",
        {
            "form": form,
            "expenses": expenses
        }
    )
# ============================================================
# REGISTRATION + USER PROFILE
# ============================================================

def registration(request):

    if request.method == "POST":

        # ====================================================
        # REGISTRATION DETAILS
        # ====================================================

        full_name = request.POST.get("name")
        username = request.POST.get("username")
        email = request.POST.get("email")
        password = request.POST.get("password")

        # ====================================================
        # USER PROFILE DETAILS
        # ====================================================

        age = request.POST.get("age")
        gender = request.POST.get("gender")
        education_level = request.POST.get("education_level")
        employment_status = request.POST.get("employment_status")

        # ====================================================
        # VALIDATION
        # ====================================================

        if not all([
            full_name,
            username,
            email,
            password,
            age,
            gender,
            education_level,
            employment_status
        ]):

            return render(
                request,
                "registration.html",
                {
                    "error": "All fields are required."
                }
            )

        # ====================================================
        # CHECK USERNAME
        # ====================================================

        if User.objects.filter(
            username=username
        ).exists():

            return render(
                request,
                "registration.html",
                {
                    "error": "Username already exists."
                }
            )

        # ====================================================
        # CHECK EMAIL
        # ====================================================

        if User.objects.filter(
            email=email
        ).exists():

            return render(
                request,
                "registration.html",
                {
                    "error": "Email already exists."
                }
            )

        # ====================================================
        # CREATE USER
        # ====================================================

        User.objects.create_user(
            username=username,
            email=email,
            password=password,
            first_name=full_name
        )

        # ====================================================
        # SAVE PROFILE INFORMATION IN SESSION
        # ====================================================

        request.session["age"] = int(age)

        request.session["gender"] = gender

        request.session["education_level"] = education_level

        request.session["employment_status"] = employment_status

        request.session.modified = True

        # ====================================================
        # REGISTRATION SUCCESS
        # ====================================================

        return redirect("login")

    # ========================================================
    # GET REQUEST
    # ========================================================

    return render(
        request,
        "registration.html"
    )





# ============================================================
# LOAN INFORMATION
# ============================================================

def loan_information(request):

    if not request.user.is_authenticated:

        return redirect(
            "login"
        )


    if request.method == "POST":

        form = LoanForm(
            request.POST
        )

        if form.is_valid():

            loan = form.save(
                commit=False
            )

            loan.user = request.user

            loan.save()

            return redirect(
                "loan_information"
            )

    else:

        form = LoanForm()


    loans = (
        Loan.objects
        .filter(
            user=request.user
        )
        .order_by(
            "-id"
        )
    )


    return render(
        request,
        "loan_information.html",
        {
            "form": form,
            "loans": loans
        }
    )


# ============================================================
# INCOME TRACKER
# ============================================================

def income_tracker(request):

    if not request.user.is_authenticated:

        return redirect(
            "login"
        )


    if request.method == "POST":

        form = IncomeForm(
            request.POST
        )

        if form.is_valid():

            income = form.save(
                commit=False
            )

            income.user = request.user

            income.save()

            return redirect(
                "income_tracker"
            )

    else:

        form = IncomeForm()


    incomes = (
        Income.objects
        .filter(
            user=request.user
        )
        .order_by(
            "-date",
            "-id"
        )
    )


    return render(
        request,
        "income_tracker.html",
        {
            "form": form,
            "incomes": incomes
        }
    )


def financial_analysis(request):

    # ========================================================
    # LOGIN CHECK
    # ========================================================

    if not request.user.is_authenticated:
        return redirect("login")


    # ========================================================
    # CHECK MODEL
    # ========================================================

    if best_model is None:

        return render(
            request,
            "financial_analysis.html",
            {
                "error":
                    "Logistic Regression model could not be loaded."
            }
        )


    # ========================================================
    # GET SELECTED MONTH
    # ========================================================

    selected_month = request.GET.get(
        "month",
        datetime.now().strftime("%Y-%m")
    )


    # ========================================================
    # CONVERT SELECTED MONTH
    # ========================================================

    try:

        selected_date = datetime.strptime(
            selected_month,
            "%Y-%m"
        )

        selected_year = selected_date.year

        selected_month_number = selected_date.month

    except ValueError:

        selected_month = datetime.now().strftime("%Y-%m")

        selected_date = datetime.strptime(
            selected_month,
            "%Y-%m"
        )

        selected_year = selected_date.year

        selected_month_number = selected_date.month


    # ========================================================
    # USER RECORDS
    # ========================================================

    income_queryset = Income.objects.filter(
        user=request.user
    )

    expense_queryset = Expense.objects.filter(
        user=request.user
    )

    loan_queryset = Loan.objects.filter(
        user=request.user
    )


    # ========================================================
    # MONTH FILTER
    # ========================================================

    income_queryset = income_queryset.filter(
        date__year=selected_year,
        date__month=selected_month_number
    )

    expense_queryset = expense_queryset.filter(
        date__year=selected_year,
        date__month=selected_month_number
    )

    loan_queryset = loan_queryset.filter(
        date__year=selected_year,
        date__month=selected_month_number
    )


    # ========================================================
    # TOTAL INCOME
    # ========================================================

    total_income = (
        income_queryset
        .aggregate(
            total=Sum("amount")
        )["total"]
        or 0
    )


    # ========================================================
    # TOTAL EXPENSES
    # ========================================================

    total_expenses = (
        expense_queryset
        .aggregate(
            total=Sum("amount")
        )["total"]
        or 0
    )


    # ========================================================
    # TOTAL LOAN
    # ========================================================

    total_loan = (
        loan_queryset
        .aggregate(
            total=Sum("amount")
        )["total"]
        or 0
    )


    # ========================================================
    # TOTAL EMI
    # ========================================================

    total_emi = (
        loan_queryset
        .aggregate(
            total=Sum("monthly_emi")
        )["total"]
        or 0
    )


    # ========================================================
    # CONVERT TO FLOAT
    # ========================================================

    total_income = float(total_income)

    total_expenses = float(total_expenses)

    total_loan = float(total_loan)

    total_emi = float(total_emi)


    # ========================================================
    # SAVINGS
    # ========================================================

    savings = (
        total_income
        - total_expenses
        - total_emi
    )


    # ========================================================
    # ML RATIOS
    # ========================================================

    if total_income > 0:

        expense_ratio_ml = (
            total_expenses /
            total_income
        )

        savings_ratio_ml = (
            savings /
            total_income
        )

        debt_to_income_ratio_ml = (
            total_emi /
            total_income
        )

    else:

        expense_ratio_ml = 0.0

        savings_ratio_ml = 0.0

        debt_to_income_ratio_ml = 0.0


    # ========================================================
    # DISPLAY RATIOS
    # ========================================================

    expense_ratio = (
        expense_ratio_ml * 100
    )

    savings_ratio = (
        savings_ratio_ml * 100
    )

    debt_to_income_ratio = (
        debt_to_income_ratio_ml * 100
    )


    # ========================================================
    # LOAN STATUS
    # ========================================================

    if total_loan > 0:

        has_loan = "Yes"

    else:

        has_loan = "No"


    # ========================================================
    # PROFILE DATA
    # ========================================================

    age = request.session.get(
        "age"
    )

    gender = request.session.get(
        "gender"
    )

    education_level = request.session.get(
        "education_level"
    )

    employment_status = request.session.get(
        "employment_status"
    )


    # ========================================================
    # CHECK PROFILE
    # ========================================================

    if not all([
        age,
        gender,
        education_level,
        employment_status
    ]):

        return render(
            request,
            "financial_analysis.html",
            {
                "error":
                    "Please complete your profile "
                    "information before generating "
                    "the financial analysis.",

                "selected_month":
                    selected_month
            }
        )


    # ========================================================
    # CONVERT AGE
    # ========================================================

    try:

        age = float(age)

    except (
        TypeError,
        ValueError
    ):

        return render(
            request,
            "financial_analysis.html",
            {
                "error":
                    "Invalid age value in profile.",

                "selected_month":
                    selected_month
            }
        )


    # ========================================================
    # FIX EMPLOYMENT SPELLING
    # ========================================================

    if employment_status == "Self-Employed":

        employment_status = "Self-employed"


    # ========================================================
    # CREATE 13 FEATURES
    # ========================================================

    feature_names = [

        "age",
        "gender",
        "education_level",
        "employment_status",
        "monthly_income_inr",
        "monthly_expenses_inr",
        "savings_inr",
        "has_loan",
        "loan_amount_inr",
        "monthly_emi_inr",
        "debt_to_income_ratio",
        "savings_to_income_ratio",
        "expense_ratio"

    ]


    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    features = pd.DataFrame(

        [[

            age,

            gender,

            education_level,

            employment_status,

            total_income,

            total_expenses,

            savings,

            has_loan,

            total_loan,

            total_emi,

            debt_to_income_ratio_ml,

            savings_ratio_ml,

            expense_ratio_ml

        ]],

        columns=feature_names

    )


    # ========================================================
    # DEBUG INPUT
    # ========================================================

    print("\n")

    print("=" * 70)

    print(
        "FINANCIAL ANALYSIS - LOGISTIC REGRESSION MODEL INPUT"
    )

    print("=" * 70)

    print(
        features.to_string(
            index=False
        )
    )

    print(
        "Number of features:",
        len(features.columns)
    )

    print(
        "Feature names:",
        list(features.columns)
    )

    print("=" * 70)


    # ========================================================
    # LOGISTIC REGRESSION PREDICTION
    # ========================================================

    try:

        prediction = best_model.predict(
            features
        )

        financial_status = str(
            prediction[0]
        )

        print(
            "Logistic Regression Prediction:",
            financial_status
        )

    except Exception as e:

        return render(
            request,
            "financial_analysis.html",
            {
                "error":
                    f"Prediction error: {str(e)}",

                "selected_month":
                    selected_month
            }
        )


    # ========================================================
    # RECOMMENDATION
    # ========================================================

    if financial_status == "Good":

        recommendation = (
            "Your financial status is predicted as Good. "
            "Your income, expenses, savings, and financial ratios "
            "show a healthy financial condition. Continue saving "
            "regularly, control unnecessary expenses, and maintain "
            "your current financial habits."
        )


    elif financial_status == "Average":

        recommendation = (
            "Your financial condition is predicted as Average. "
            "Try to improve your savings and reduce unnecessary "
            "expenses to achieve better financial stability."
        )


    elif financial_status == "Poor":

        recommendation = (
            "Your financial status is predicted as Poor. "
            "Your income, expenses, savings, and financial ratios "
            "indicate that your financial condition needs attention. "
            "Try to reduce unnecessary expenses, increase your savings, "
            "and improve your overall financial management."
        )


    else:

        recommendation = (
            "Financial prediction was generated "
            "by the trained Logistic Regression model."
        )


    # ========================================================
    # RISK MANAGEMENT
    # ========================================================

    if financial_status == "Good":

        risk_management = [

            "Maintain an emergency fund",

            "Continue keeping expenses under control",

            "Avoid unnecessary debt",

            "Keep monthly EMI manageable",

            "Continue regular savings"

        ]


    elif financial_status == "Average":

        risk_management = [

            "Maintain an emergency fund",

            "Monitor unnecessary expenses",

            "Carefully manage loan payments",

            "Keep EMI payments manageable",

            "Try to increase monthly savings"

        ]


    elif financial_status == "Poor":

        risk_management = [

            "Reduce unnecessary expenses",

            "Avoid taking unnecessary loans",

            "Carefully manage EMI payments",

            "Increase emergency savings",

            "Review monthly spending regularly"

        ]


    else:

        risk_management = [

            "Monitor monthly expenses",

            "Maintain emergency savings",

            "Review loan commitments regularly"

        ]


    # ========================================================
    # CONTEXT
    # ========================================================

    context = {

        "total_income":
            round(
                total_income,
                2
            ),

        "total_expenses":
            round(
                total_expenses,
                2
            ),

        "total_loan":
            round(
                total_loan,
                2
            ),

        "total_emi":
            round(
                total_emi,
                2
            ),

        "savings":
            round(
                savings,
                2
            ),

        "expense_ratio":
            round(
                expense_ratio,
                2
            ),

        "savings_ratio":
            round(
                savings_ratio,
                2
            ),

        "debt_to_income_ratio":
            round(
                debt_to_income_ratio,
                2
            ),

        "loan_value":
            has_loan,

        "age":
            age,

        "gender":
            gender,

        "education_level":
            education_level,

        "employment_status":
            employment_status,

        "selected_month":
            selected_month,

        "financial_status":
            financial_status,

        "recommendation":
            recommendation,

        "risk_management":
            risk_management,

        "error":
            None

    }


    # ========================================================
    # DEBUG RESULT
    # ========================================================

    print("\n")

    print("=" * 70)

    print(
        "FINAL FINANCIAL ANALYSIS"
    )

    print("=" * 70)

    print(
        "Financial Status:",
        financial_status
    )

    print(
        "Income:",
        total_income
    )

    print(
        "Expenses:",
        total_expenses
    )

    print(
        "Loan:",
        total_loan
    )

    print(
        "EMI:",
        total_emi
    )

    print(
        "Savings:",
        savings
    )

    print(
        "Expense Ratio:",
        expense_ratio,
        "%"
    )

    print(
        "Savings Ratio:",
        savings_ratio,
        "%"
    )

    print(
        "Debt Ratio:",
        debt_to_income_ratio,
        "%"
    )

    print("=" * 70)


    # ========================================================
    # SHOW ANALYSIS
    # ========================================================

    return render(
        request,
        "financial_analysis.html",
        context
    )
def financial_details(request):

    if not request.user.is_authenticated:
        return redirect('login')


    # =====================================================
    # INCOME FORM
    # =====================================================

    income_form = IncomeForm()


    # =====================================================
    # EXPENSE FORM
    # =====================================================

    expense_form = ExpenseForm()


    # =====================================================
    # LOAN FORM
    # =====================================================

    loan_form = LoanForm()


    # =====================================================
    # HANDLE POST
    # =====================================================

    if request.method == 'POST':

        form_type = request.POST.get('form_type')


        # -------------------------------------------------
        # INCOME
        # -------------------------------------------------

        if form_type == 'income':

            income_form = IncomeForm(request.POST)

            if income_form.is_valid():

                income = income_form.save(commit=False)

                income.user = request.user

                income.save()

                return redirect('financial_details')


        # -------------------------------------------------
        # EXPENSE
        # -------------------------------------------------

        elif form_type == 'expense':

            expense_form = ExpenseForm(request.POST)

            if expense_form.is_valid():

                expense = expense_form.save(commit=False)

                expense.user = request.user

                expense.save()

                return redirect('financial_details')


        # -------------------------------------------------
        # LOAN
        # -------------------------------------------------

        elif form_type == 'loan':

            loan_form = LoanForm(request.POST)

            if loan_form.is_valid():

                loan = loan_form.save(commit=False)

                loan.user = request.user

                loan.save()

                return redirect('financial_details')


    # =====================================================
    # FETCH ONLY THIS USER'S DATA
    # =====================================================

    incomes = Income.objects.filter(
        user=request.user
    ).order_by('-date', '-id')


    expenses = Expense.objects.filter(
        user=request.user
    ).order_by('-date', '-id')


    loans = Loan.objects.filter(
        user=request.user
    ).order_by('-date', '-id')


    # =====================================================
    # PAGE
    # =====================================================

    return render(
        request,
        'financial_details.html',
        {
            'income_form': income_form,
            'expense_form': expense_form,
            'loan_form': loan_form,

            'incomes': incomes,
            'expenses': expenses,
            'loans': loans,
        }
    ) 

# =========================================================
# EDIT INCOME
# =========================================================

def edit_income(request, id):

    income = get_object_or_404(
        Income,
        id=id,
        user=request.user
    )

    if request.method == "POST":

        income.source = request.POST.get("source")
        income.amount = request.POST.get("amount")
        income.date = request.POST.get("date")

        income.save()

        return redirect("financial_details")

    return render(
        request,
        "edit_income.html",
        {"income": income}
    )


def delete_income(request, id):
    income = get_object_or_404(
        Income,
        id=id,
        user=request.user
    )

    income.delete()

    return redirect("financial_details")

# =========================================================
# EDIT EXPENSE
# =========================================================

def edit_expense(request, id):

    expense = get_object_or_404(
        Expense,
        id=id,
        user=request.user
    )

    if request.method == "POST":

        expense.category = request.POST.get("category")
        expense.amount = request.POST.get("amount")
        expense.date = request.POST.get("date")

        expense.save()

        return redirect("financial_details")

    return render(
        request,
        "edit_expense.html",
        {"expense": expense}
    )


def delete_expense(request, id):

    expense = get_object_or_404(
        Expense,
        id=id,
        user=request.user
    )

    expense.delete()

    return redirect("financial_details")

# =========================================================
# EDIT LOAN
# =========================================================

def edit_loan(request, id):

    loan = get_object_or_404(
        Loan,
        id=id,
        user=request.user
    )

    if request.method == "POST":

        loan.loan_type = request.POST.get("loan_type")
        loan.amount = request.POST.get("amount")
        loan.monthly_emi = request.POST.get("monthly_emi")
        loan.date = request.POST.get("date")

        loan.save()

        return redirect("financial_details")

    return render(
        request,
        "edit_loan.html",
        {"loan": loan}
    )


def delete_loan(request, id):

    loan = get_object_or_404(
        Loan,
        id=id,
        user=request.user
    )

    loan.delete()

    return redirect("financial_details")


def dashboard(request):

    # =========================================================
    # LOGIN CHECK
    # =========================================================

    if not request.user.is_authenticated:
        return redirect("login")

    user = request.user

    # =========================================================
    # 1. FETCH ALL INCOME DATA
    # =========================================================

    income_records = (
        Income.objects
        .filter(user=user)
        .order_by("date")
    )

    # =========================================================
    # 2. FETCH ALL EXPENSE DATA
    # =========================================================

    expense_records = (
        Expense.objects
        .filter(user=user)
        .order_by("date")
    )

    # =========================================================
    # 3. FETCH ALL LOAN DATA
    # =========================================================

    loan_records = (
        Loan.objects
        .filter(user=user)
        .order_by("date")
    )

    # =========================================================
    # 4. TOTAL INCOME
    # =========================================================

    total_income = (
        income_records.aggregate(
            total=Sum("amount")
        )["total"] or 0
    )

    # =========================================================
    # 5. TOTAL EXPENSE
    # =========================================================

    total_expenses = (
        expense_records.aggregate(
            total=Sum("amount")
        )["total"] or 0
    )

    # =========================================================
    # 6. TOTAL EMI
    # =========================================================

    total_emi = (
        loan_records.aggregate(
            total=Sum("monthly_emi")
        )["total"] or 0
    )

    # =========================================================
    # 7. TOTAL LOAN AMOUNT
    # =========================================================

    total_loan = (
        loan_records.aggregate(
            total=Sum("amount")
        )["total"] or 0
    )

    # =========================================================
    # 8. CALCULATE SAVINGS
    # =========================================================

    savings = (
        total_income
        - total_expenses
        - total_emi
    )

    # =========================================================
    # 9. MONTHLY INCOME
    # =========================================================

    income_data = (
        income_records
        .annotate(month=TruncMonth("date"))
        .values("month")
        .annotate(total=Sum("amount"))
        .order_by("month")
    )

    # =========================================================
    # 10. MONTHLY EXPENSE
    # =========================================================

    expense_data = (
        expense_records
        .annotate(month=TruncMonth("date"))
        .values("month")
        .annotate(total=Sum("amount"))
        .order_by("month")
    )

    # =========================================================
    # 11. MONTHLY EMI
    # =========================================================

    loan_data = (
        loan_records
        .annotate(month=TruncMonth("date"))
        .values("month")
        .annotate(total=Sum("monthly_emi"))
        .order_by("month")
    )

    # =========================================================
    # 12. CREATE DICTIONARIES
    # =========================================================

    income_dict = {
        item["month"].strftime("%Y-%m"):
        float(item["total"] or 0)
        for item in income_data
        if item["month"]
    }

    expense_dict = {
        item["month"].strftime("%Y-%m"):
        float(item["total"] or 0)
        for item in expense_data
        if item["month"]
    }

    emi_dict = {
        item["month"].strftime("%Y-%m"):
        float(item["total"] or 0)
        for item in loan_data
        if item["month"]
    }

    # =========================================================
    # 13. GET ALL AVAILABLE MONTHS
    # =========================================================

    months = sorted(
        set(income_dict.keys())
        | set(expense_dict.keys())
        | set(emi_dict.keys())
    )

    # Only latest 6 months
    months = months[-6:]

    # =========================================================
    # 14. CHART VALUES
    # =========================================================

    income_values = [
        income_dict.get(month, 0)
        for month in months
    ]

    expense_values = [
        expense_dict.get(month, 0)
        for month in months
    ]

    emi_values = [
        emi_dict.get(month, 0)
        for month in months
    ]

    # =========================================================
    # 15. SAVINGS FOR EACH MONTH
    # =========================================================

    savings_values = [
        income - expense - emi
        for income, expense, emi in zip(
            income_values,
            expense_values,
            emi_values
        )
    ]

    # =========================================================
    # 16. DISPLAY MONTHS
    # =========================================================

    from datetime import datetime

    display_months = []

    for month in months:

        date_object = datetime.strptime(
            month,
            "%Y-%m"
        )

        display_months.append(
            date_object.strftime("%b %Y")
        )

    # =========================================================
    # 17. EXPENSE CATEGORY DATA
    # =========================================================

    category_data = (
        expense_records
        .values("category")
        .annotate(total=Sum("amount"))
        .order_by("-total")
    )

    category_names = [
        item["category"] or "Other"
        for item in category_data
    ]

    category_values = [
        float(item["total"] or 0)
        for item in category_data
    ]

    # =========================================================
    # 18. SEND DATA TO DASHBOARD
    # =========================================================

    context = {

        # Summary
        "total_income": total_income,
        "total_expenses": total_expenses,
        "total_emi": total_emi,
        "total_loan": total_loan,
        "savings": savings,

        # Monthly chart
        "months": json.dumps(display_months),
        "income_values": json.dumps(income_values),
        "expense_values": json.dumps(expense_values),
        "savings_values": json.dumps(savings_values),
        "emi_values": json.dumps(emi_values),

        # Expense chart
        "category_names": json.dumps(category_names),
        "category_values": json.dumps(category_values),

        # Optional: actual records
        "income_records": income_records,
        "expense_records": expense_records,
        "loan_records": loan_records,
    }

    return render(
        request,
        "dashboard.html",
        context
    )
