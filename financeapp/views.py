import pandas as pd
import joblib
import os

from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth.hashers import make_password
from django.db.models import Sum
from django.utils import timezone
from django.contrib.auth import authenticate, login, logout
from .models import Income, Expense, Loan
from .forms import IncomeForm, ExpenseForm, LoanForm
from django.conf import settings

SVM_MODEL_PATH = os.path.join(
    settings.BASE_DIR,
    "svm_model.pkl"
)

SVM_SCALER_PATH = os.path.join(
    settings.BASE_DIR,
    "svm_scaler.pkl"
)

SVM_ENCODERS_PATH = os.path.join(
    settings.BASE_DIR,
    "svm_encoders.pkl"
)

FEATURES_PATH = os.path.join(
    settings.BASE_DIR,
    "selected_features.pkl"
)


CREDIT_MODEL_PATH = os.path.join(
    settings.BASE_DIR,
    "credit_score_model.pkl"
)

CREDIT_SCALER_PATH = os.path.join(
    settings.BASE_DIR,
    "credit_score_scaler.pkl"
)

CREDIT_LOAN_ENCODER_PATH = os.path.join(
    settings.BASE_DIR,
    "credit_score_loan_encoder.pkl"
)




# Load SVM model
svm_model = joblib.load(
    SVM_MODEL_PATH
)


# Load scaler
svm_scaler = joblib.load(
    SVM_SCALER_PATH
)


# Load all encoders
svm_encoders = joblib.load(
    SVM_ENCODERS_PATH
)


# Load exact feature order
selected_features = joblib.load(
    FEATURES_PATH
)


# Get individual encoders
gender_encoder = svm_encoders["gender"]

education_encoder = svm_encoders["education_level"]

employment_encoder = svm_encoders["employment_status"]

loan_encoder = svm_encoders["has_loan"]

target_encoder = svm_encoders["target"]

def home(request):

    return render(
        request,
        "home.html"
    )

# =========================================================
# LOGIN
# =========================================================

def login_view(request):

    if request.method == "POST":

        username = request.POST.get("username")
        password = request.POST.get("password")

        user = authenticate(
            request,
            username=username,
            password=password
        )

        if user is not None:

            login(request, user)

            return redirect("user_profile")

        return render(
            request,
            "login.html",
            {
                "error": "Invalid username or password"
            }
        )

    return render(request, "login.html")

def registration(request):

    if request.method == 'POST':

        # Get values from registration form
        full_name = request.POST.get('name')
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')

        # Check whether all fields are entered
        if not all([
            full_name,
            username,
            email,
            password
        ]):
            return render(
                request,
                'registration.html',
                {
                    'error': 'All fields are required.'
                }
            )

        # Check whether username already exists
        if User.objects.filter(username=username).exists():

            return render(
                request,
                'registration.html',
                {
                    'error': 'Username already exists.'
                }
            )

        # Check whether email already exists
        if User.objects.filter(email=email).exists():

            return render(
                request,
                'registration.html',
                {
                    'error': 'Email already exists.'
                }
            )

        # Create Django user
        User.objects.create_user(
            username=username,
            email=email,
            password=password,
            first_name=full_name
        )

        # Registration successful
        return redirect('login')

    return render(
        request,
        'registration.html'
    )




def expense_tracker(request):

    if request.method == 'POST':

        form = ExpenseForm(request.POST)

        if form.is_valid():

            expense = form.save(commit=False)

            expense.user = request.user

            expense.save()

            return redirect('expense_tracker')

    else:

        form = ExpenseForm()

    expenses = Expense.objects.filter(
        user=request.user
    ).order_by('-date', '-id')

    return render(
        request,
        'expense_tracker.html',
        {
            'form': form,
            'expenses': expenses
        }
    )

def financial_analysis(request):

    if not request.user.is_authenticated:
        return redirect('login')

    # =========================================================
    # 1. TOTAL INCOME
    # =========================================================

    total_income = Income.objects.filter(
        user=request.user
    ).aggregate(
        total=Sum('amount')
    )['total'] or 0

    # =========================================================
    # 2. TOTAL EXPENSES
    # =========================================================

    total_expenses = Expense.objects.filter(
        user=request.user
    ).aggregate(
        total=Sum('amount')
    )['total'] or 0

    # =========================================================
    # 3. TOTAL LOAN
    # =========================================================

    total_loan = Loan.objects.filter(
        user=request.user
    ).aggregate(
        total=Sum('amount')
    )['total'] or 0

    # =========================================================
    # 4. TOTAL EMI
    # =========================================================

    total_emi = Loan.objects.filter(
        user=request.user
    ).aggregate(
        total=Sum('monthly_emi')
    )['total'] or 0

    # =========================================================
    # 5. CONVERT TO FLOAT
    # =========================================================

    total_income = float(total_income)
    total_expenses = float(total_expenses)
    total_loan = float(total_loan)
    total_emi = float(total_emi)

    # =========================================================
    # 6. SAVINGS
    # =========================================================

    savings = total_income - total_expenses - total_emi

    # =========================================================
    # 7. ML RATIOS
    # IMPORTANT:
    # SVM DATASET USES DECIMAL RATIOS
    # NOT PERCENTAGES
    # =========================================================

    if total_income > 0:

        expense_ratio_ml = (
            total_expenses / total_income
        )

        savings_ratio_ml = (
            savings / total_income
        )

        debt_to_income_ratio_ml = (
            total_emi / total_income
        )

    else:

        expense_ratio_ml = 0.0
        savings_ratio_ml = 0.0
        debt_to_income_ratio_ml = 0.0

    # =========================================================
    # 8. DISPLAY RATIOS
    # =========================================================

    expense_ratio = expense_ratio_ml * 100
    savings_ratio = savings_ratio_ml * 100
    debt_to_income_ratio = debt_to_income_ratio_ml * 100

    # =========================================================
    # 9. LOAN STATUS
    # =========================================================

    if total_loan > 0:

        loan_value = "Yes"

    else:

        loan_value = "No"

    # =========================================================
    # 10. USER PROFILE
    # =========================================================

    age = request.session.get(
        'age',
        25
    )

    gender = request.session.get(
        'gender',
        'Other'
    )

    education_level = request.session.get(
        'education_level',
        'Other'
    )

    employment_status = request.session.get(
        'employment_status',
        'Employed'
    )

    # Fix spelling used in dataset
    if employment_status == "Self-Employed":
        employment_status = "Self-employed"

    # =========================================================
    # 11. AUTOMATIC CREDIT SCORE
    # =========================================================

    credit_score = None

    try:

        # -----------------------------------------------------
        # LOAD MODEL
        # -----------------------------------------------------

        credit_model = joblib.load(
            CREDIT_MODEL_PATH
        )

        credit_scaler = joblib.load(
            CREDIT_SCALER_PATH
        )

        credit_loan_encoder = joblib.load(
            CREDIT_LOAN_ENCODER_PATH
        )

        # -----------------------------------------------------
        # ENCODE LOAN
        # -----------------------------------------------------

        has_loan_encoded = credit_loan_encoder.transform(
            [loan_value]
        )[0]

        # -----------------------------------------------------
        # CREDIT SCORE FEATURES
        # SAME ORDER AS TRAINING
        # -----------------------------------------------------

        credit_features = [
            "monthly_income_usd",
            "monthly_expenses_usd",
            "savings_usd",
            "has_loan",
            "loan_amount_usd",
            "monthly_emi_usd",
            "debt_to_income_ratio",
            "savings_to_income_ratio",
            "expense_ratio"
        ]

        # -----------------------------------------------------
        # CREATE CREDIT MODEL INPUT
        # -----------------------------------------------------

        credit_input = pd.DataFrame(
            [[
                total_income,
                total_expenses,
                savings,
                has_loan_encoded,
                total_loan,
                total_emi,
                debt_to_income_ratio_ml,
                savings_ratio_ml,
                expense_ratio_ml
            ]],
            columns=credit_features
        )

        # -----------------------------------------------------
        # DEBUG
        # -----------------------------------------------------

        print("\n")
        print("=" * 70)
        print("AUTOMATIC CREDIT SCORE INPUT")
        print("=" * 70)
        print(credit_input)
        print("=" * 70)

        # -----------------------------------------------------
        # SCALE
        # -----------------------------------------------------

        scaled_credit_input = credit_scaler.transform(
            credit_input
        )

        # -----------------------------------------------------
        # PREDICT
        # -----------------------------------------------------

        predicted_score = credit_model.predict(
            scaled_credit_input
        )[0]

        credit_score = round(
            float(predicted_score)
        )

        # -----------------------------------------------------
        # LIMIT TO STANDARD DATASET RANGE
        # -----------------------------------------------------

        credit_score = max(
            300,
            min(850, credit_score)
        )

        # -----------------------------------------------------
        # SAVE TO SESSION
        # -----------------------------------------------------

        request.session["credit_score"] = credit_score
        request.session.modified = True

        print("AUTOMATIC CREDIT SCORE")
        print("Predicted Score:", credit_score)
        print("=" * 70)

    except Exception as e:

        print("\n")
        print("=" * 70)
        print("CREDIT SCORE ERROR")
        print("Error Type:", type(e).__name__)
        print("Error:", str(e))
        print("=" * 70)

        credit_score = None

    # =========================================================
    # 12. SEND DATA TO HTML
    # =========================================================

    context = {

        "total_income":
            round(total_income, 2),

        "total_expenses":
            round(total_expenses, 2),

        "total_loan":
            round(total_loan, 2),

        "total_emi":
            round(total_emi, 2),

        "savings":
            round(savings, 2),

        # Display percentages
        "expense_ratio":
            round(expense_ratio, 2),

        "savings_ratio":
            round(savings_ratio, 2),

        "debt_to_income_ratio":
            round(debt_to_income_ratio, 2),

        # ML generated credit score
        "credit_score":
            credit_score,

        "age":
            age,

        "gender":
            gender,

        "education_level":
            education_level,

        "employment_status":
            employment_status,

    }

    # =========================================================
    # 13. RETURN PAGE
    # =========================================================

    return render(
        request,
        "financial_analysis.html",
        context
    )

def monthly_report(request):

    # -----------------------------------------------------
    # 1. CHECK LOGIN
    # -----------------------------------------------------

    if not request.user.is_authenticated:
        return redirect('login')


    # -----------------------------------------------------
    # 2. THIS USER'S INCOME
    # -----------------------------------------------------

    total_income = Income.objects.filter(
        user=request.user
    ).aggregate(
        total=Sum('amount')
    )['total'] or 0


    # -----------------------------------------------------
    # 3. THIS USER'S EXPENSES
    # -----------------------------------------------------

    total_expenses = Expense.objects.filter(
        user=request.user
    ).aggregate(
        total=Sum('amount')
    )['total'] or 0


    # Convert to float
    total_income = float(total_income)
    total_expenses = float(total_expenses)


    # -----------------------------------------------------
    # 4. CALCULATE SAVINGS
    # -----------------------------------------------------

    total_savings = total_income - total_expenses


    # -----------------------------------------------------
    # 5. CALCULATE RATIOS
    # -----------------------------------------------------

    if total_income > 0:

        savings_ratio = (
            total_savings / total_income
        ) * 100

        expense_ratio = (
            total_expenses / total_income
        ) * 100

    else:

        savings_ratio = 0
        expense_ratio = 0


    # -----------------------------------------------------
    # 6. EXPENSE CATEGORY BREAKDOWN
    # -----------------------------------------------------

    expense_categories = Expense.objects.filter(
        user=request.user
    ).values(
        'category'
    ).annotate(
        total=Sum('amount')
    ).order_by('-total')


    # -----------------------------------------------------
    # 7. HIGHEST EXPENSE CATEGORY
    # -----------------------------------------------------

    highest_expense = expense_categories.first()


    # -----------------------------------------------------
    # 8. SEND DATA TO HTML
    # -----------------------------------------------------

    context = {

        'total_income': total_income,

        'total_expenses': total_expenses,

        'total_savings': total_savings,

        'savings_ratio': round(
            savings_ratio,
            2
        ),

        'expense_ratio': round(
            expense_ratio,
            2
        ),

        'expense_categories': expense_categories,

        'highest_expense': highest_expense,

    }


    return render(
        request,
        'monthly_report.html',
        context
    )



def user_profile(request):

    if not request.user.is_authenticated:
        return redirect('login')

    if request.method == 'POST':

        age = request.POST.get('age')
        gender = request.POST.get('gender')
        education_level = request.POST.get('education_level')
        employment_status = request.POST.get('employment_status')

        if not all([
            age,
            gender,
            education_level,
            employment_status
        ]):
            return render(
                request,
                'user_profile.html',
                {
                    'error': 'All fields are required.'
                }
            )

        request.session['age'] = int(age)
        request.session['gender'] = gender
        request.session['education_level'] = education_level
        request.session['employment_status'] = employment_status

        return redirect('income_tracker')

    return render(
        request,
        'user_profile.html'
    )





def loan_information(request):
    if request.method == 'POST':
        form = LoanForm(request.POST)

        if form.is_valid():
            loan = form.save(commit=False)
            loan.user = request.user
            loan.save()
            return redirect('loan_information')
    else:
        form = LoanForm()

    loans = Loan.objects.filter(user=request.user).order_by('-id')

    return render(
        request,
        'loan_information.html',
        {
            'form': form,
            'loans': loans
        }
    )
    
def financial_report(request):

    # ==========================================================
    # LOGIN CHECK
    # ==========================================================

    if not request.user.is_authenticated:
        return redirect("login")


    # ==========================================================
    # LOAD SVM FILES
    # ==========================================================

    try:
        svm_model = joblib.load(SVM_MODEL_PATH)
        svm_scaler = joblib.load(SVM_SCALER_PATH)
        svm_encoders = joblib.load(SVM_ENCODERS_PATH)

    except Exception as e:

        return render(
            request,
            "financial_report.html",
            {
                "error": f"SVM files could not be loaded: {str(e)}"
            }
        )


    # ==========================================================
    # GET PROFILE DATA FROM SESSION
    # ==========================================================

    age = request.session.get("age")
    gender = request.session.get("gender")
    education_level = request.session.get("education_level")
    employment_status = request.session.get("employment_status")


    if age is None:

        return render(
            request,
            "financial_report.html",
            {
                "error": "Age is missing. Please complete your profile."
            }
        )


    if gender is None:

        return render(
            request,
            "financial_report.html",
            {
                "error": "Gender is missing. Please complete your profile."
            }
        )


    if education_level is None:

        return render(
            request,
            "financial_report.html",
            {
                "error": "Education level is missing. Please complete your profile."
            }
        )


    if employment_status is None:

        return render(
            request,
            "financial_report.html",
            {
                "error": "Employment status is missing. Please complete your profile."
            }
        )


    # ==========================================================
    # GET FINANCIAL DATA FROM DATABASE
    # ==========================================================

    total_income = (
        Income.objects
        .filter(user=request.user)
        .aggregate(total=Sum("amount"))["total"] or 0
    )


    total_expenses = (
        Expense.objects
        .filter(user=request.user)
        .aggregate(total=Sum("amount"))["total"] or 0
    )


    total_loan = (
        Loan.objects
        .filter(user=request.user)
        .aggregate(total=Sum("amount"))["total"] or 0
    )


    total_emi = (
        Loan.objects
        .filter(user=request.user)
        .aggregate(total=Sum("monthly_emi"))["total"] or 0
    )


    # Convert Decimal values to float

    total_income = float(total_income)
    total_expenses = float(total_expenses)
    total_loan = float(total_loan)
    total_emi = float(total_emi)


    # ==========================================================
    # CALCULATE SAVINGS
    # ==========================================================

    savings = total_income - total_expenses - total_emi


    # ==========================================================
    # CALCULATE RATIOS
    #
    # IMPORTANT:
    # Keep them as decimal values for SVM.
    #
    # Example:
    # 40% = 0.40
    # ==========================================================

    if total_income > 0:

        expense_ratio = total_expenses / total_income

        savings_to_income_ratio = savings / total_income

        debt_to_income_ratio = total_emi / total_income

    else:

        expense_ratio = 0

        savings_to_income_ratio = 0

        debt_to_income_ratio = 0


    # ==========================================================
    # ENCODE GENDER
    # ==========================================================

    try:

        gender_encoder = svm_encoders["gender"]

        gender_encoded = gender_encoder.transform(
            [gender]
        )[0]

    except Exception as e:

        return render(
            request,
            "financial_report.html",
            {
                "error": f"Gender encoding error: {str(e)}"
            }
        )


    # ==========================================================
    # ENCODE EDUCATION
    # ==========================================================

    try:

        education_encoder = svm_encoders["education_level"]

        education_encoded = education_encoder.transform(
            [education_level]
        )[0]

    except Exception as e:

        return render(
            request,
            "financial_report.html",
            {
                "error": f"Education encoding error: {str(e)}"
            }
        )


    # ==========================================================
    # ENCODE EMPLOYMENT
    # ==========================================================

    try:

        employment_encoder = svm_encoders["employment_status"]

        employment_encoded = employment_encoder.transform(
            [employment_status]
        )[0]

    except Exception as e:

        return render(
            request,
            "financial_report.html",
            {
                "error": f"Employment encoding error: {str(e)}"
            }
        )


    # ==========================================================
    # LOAN STATUS
    # ==========================================================

    if total_loan > 0:

        has_loan = "Yes"

    else:

        has_loan = "No"


    # ==========================================================
    # ENCODE LOAN
    # ==========================================================

    try:

        loan_encoder = svm_encoders["has_loan"]

        loan_encoded = loan_encoder.transform(
            [has_loan]
        )[0]

    except Exception as e:

        return render(
            request,
            "financial_report.html",
            {
                "error": f"Loan encoding error: {str(e)}"
            }
        )


    # ==========================================================
    # GET AUTOMATIC CREDIT SCORE
    # ==========================================================

    credit_score = request.session.get("credit_score")


    if credit_score is None:

        return render(
            request,
            "financial_report.html",
            {
                "error":
                "Automatic credit score was not generated. "
                "Please complete the credit-score generation step first."
            }
        )


    try:

        credit_score = float(credit_score)

    except Exception:

        return render(
            request,
            "financial_report.html",
            {
                "error":
                "The generated credit score is not a valid number."
            }
        )


    # Keep credit score inside normal dataset range

    credit_score = max(300, min(850, credit_score))

    credit_score = round(credit_score, 2)


    # ==========================================================
    # 14 SVM FEATURES
    # ==========================================================

    feature_names = [

        "age",

        "gender",

        "education_level",

        "employment_status",

        "monthly_income_usd",

        "monthly_expenses_usd",

        "savings_usd",

        "has_loan",

        "loan_amount_usd",

        "monthly_emi_usd",

        "debt_to_income_ratio",

        "credit_score",

        "savings_to_income_ratio",

        "expense_ratio"

    ]


    # ==========================================================
    # CREATE INPUT DATAFRAME
    # ==========================================================

    features = pd.DataFrame(

        [[

            float(age),

            gender_encoded,

            education_encoded,

            employment_encoded,

            total_income,

            total_expenses,

            savings,

            loan_encoded,

            total_loan,

            total_emi,

            debt_to_income_ratio,

            credit_score,

            savings_to_income_ratio,

            expense_ratio

        ]],

        columns=feature_names

    )


    # ==========================================================
    # TERMINAL DEBUG INFORMATION
    # ==========================================================

    print("\n")
    print("=" * 70)
    print("SVM INPUT DATA")
    print("=" * 70)

    print(features.to_string(index=False))

    print("-" * 70)

    print("Credit Score:", credit_score)

    print("Expense Ratio:", expense_ratio)

    print("Savings Ratio:", savings_to_income_ratio)

    print("Debt-to-Income Ratio:", debt_to_income_ratio)

    print("Number of Features:", len(features.columns))

    print("=" * 70)


    # ==========================================================
    # CHECK FEATURE COUNT
    # ==========================================================

    if len(features.columns) != 14:

        return render(
            request,
            "financial_report.html",
            {
                "error":
                f"SVM requires 14 features, "
                f"but {len(features.columns)} were created."
            }
        )


    # ==========================================================
    # SCALE FEATURES
    # ==========================================================

    try:

        scaled_features = svm_scaler.transform(features)

    except Exception as e:

        return render(
            request,
            "financial_report.html",
            {
                "error":
                f"SVM scaling error: {str(e)}"
            }
        )


    # ==========================================================
    # SVM PREDICTION
    # ==========================================================

    try:

        prediction = svm_model.predict(
            scaled_features
        )

    except Exception as e:

        return render(
            request,
            "financial_report.html",
            {
                "error":
                f"SVM prediction error: {str(e)}"
            }
        )


    # ==========================================================
    # SHOW RAW PREDICTION IN TERMINAL
    # ==========================================================

    print("\n")
    print("=" * 70)
    print("RAW SVM PREDICTION")
    print("=" * 70)

    print(prediction)

    print("=" * 70)


    # ==========================================================
    # CONVERT PREDICTION TO GOOD / AVERAGE / POOR
    # ==========================================================

    try:

        target_encoder = svm_encoders["target"]

        financial_status = target_encoder.inverse_transform(
            prediction
        )[0]

    except Exception as e:

        return render(
            request,
            "financial_report.html",
            {
                "error":
                f"Target decoding error: {str(e)}"
            }
        )


    print("\n")
    print("=" * 70)
    print("FINAL FINANCIAL HEALTH")
    print("=" * 70)

    print(financial_status)

    print("=" * 70)


    # ==========================================================
    # RECOMMENDATION
    #
    # This is NOT used for prediction.
    # Prediction has already been done by SVM.
    # ==========================================================

    if financial_status == "Good":

        recommendation = (
            "Your financial condition is healthy. "
            "Continue maintaining your savings "
            "and controlled spending."
        )

        risk_management = (
            "Maintain an emergency fund, avoid unnecessary "
            "debt and continue regular saving."
        )


    elif financial_status == "Average":

        recommendation = (
            "Your financial condition is moderate. "
            "Try to increase savings and reduce "
            "unnecessary expenses."
        )

        risk_management = (
            "Monitor monthly expenses, control debt "
            "and maintain an emergency fund."
        )


    else:

        recommendation = (
            "Your financial condition needs attention. "
            "Try to reduce unnecessary expenses "
            "and increase savings."
        )

        risk_management = (
            "Avoid unnecessary borrowing, reduce debt "
            "burden and create an emergency fund."
        )


    # ==========================================================
    # DISPLAY RATIOS AS PERCENTAGE
    #
    # Only for HTML display.
    # SVM received decimal ratios above.
    # ==========================================================

    display_expense_ratio = round(
        expense_ratio * 100,
        2
    )

    display_savings_ratio = round(
        savings_to_income_ratio * 100,
        2
    )

    display_debt_ratio = round(
        debt_to_income_ratio * 100,
        2
    )


    # ==========================================================
    # SEND DATA TO HTML
    # ==========================================================

    context = {

        "total_income":
            round(total_income, 2),

        "total_expenses":
            round(total_expenses, 2),

        "total_loan":
            round(total_loan, 2),

        "total_emi":
            round(total_emi, 2),

        "savings":
            round(savings, 2),

        "credit_score":
            credit_score,

        "expense_ratio":
            display_expense_ratio,

        "savings_ratio":
            display_savings_ratio,

        "debt_to_income_ratio":
            display_debt_ratio,

        "financial_status":
            financial_status,

        "recommendation":
            recommendation,

        "risk_management":
            risk_management,

        "age":
            age,

        "gender":
            gender,

        "education_level":
            education_level,

        "employment_status":
            employment_status,

    }


    # ==========================================================
    # DISPLAY FINANCIAL REPORT
    # ==========================================================

    return render(
        request,
        "financial_report.html",
        context
    )

def income_tracker(request):

    if request.method == 'POST':

        form = IncomeForm(request.POST)

        if form.is_valid():

            income = form.save(commit=False)
            income.user = request.user
            income.save()

            return redirect('income_tracker')

    else:
        form = IncomeForm()

    incomes = Income.objects.filter(
        user=request.user
    ).order_by('-date', '-id')

    return render(
        request,
        'income_tracker.html',
        {
            'form': form,
            'incomes': incomes
        }
    )    