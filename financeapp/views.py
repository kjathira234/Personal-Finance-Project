import pandas as pd
import joblib
from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth.hashers import make_password

def prediction(request):

    result = None

    if request.method == "POST":

        # =========================================
        # LOAD ENCODERS
        # =========================================

        gender_encoder = joblib.load(
            "gender_encoder.pkl"
        )

        education_encoder = joblib.load(
            "education_encoder.pkl"
        )

        employment_encoder = joblib.load(
            "employment_encoder.pkl"
        )

        loan_encoder = joblib.load(
            "loan_encoder.pkl"
        )

        target_encoder = joblib.load(
            "target_encoder.pkl"
        )

        # =========================================
        # GET FORM VALUES
        # =========================================

        gender = request.POST["gender"]
        education_level = request.POST["education_level"]
        employment_status = request.POST["employment_status"]
        has_loan = request.POST["has_loan"]

        # =========================================
        # ENCODE CATEGORICAL VALUES
        # =========================================

        gender_encoded = gender_encoder.transform(
            [gender]
        )[0]

        education_encoded = education_encoder.transform(
            [education_level]
        )[0]

        employment_encoded = employment_encoder.transform(
            [employment_status]
        )[0]

        loan_encoded = loan_encoder.transform(
            [has_loan]
        )[0]

        # =========================================
        # CREATE USER DATA
        # =========================================

        user_data = pd.DataFrame([{
            "age": int(request.POST["age"]),

            "gender": gender_encoded,

            "education_level": education_encoded,

            "employment_status": employment_encoded,

            "monthly_income_usd": float(
                request.POST["monthly_income_usd"]
            ),

            "monthly_expenses_usd": float(
                request.POST["monthly_expenses_usd"]
            ),

            "savings_usd": float(
                request.POST["savings_usd"]
            ),

            "has_loan": loan_encoded,

            "loan_amount_usd": float(
                request.POST["loan_amount_usd"]
            ),

            "monthly_emi_usd": float(
                request.POST["monthly_emi_usd"]
            ),

            "debt_to_income_ratio": float(
                request.POST["debt_to_income_ratio"]
            ),

            "credit_score": float(
                request.POST["credit_score"]
            ),

            "savings_to_income_ratio": float(
                request.POST["savings_to_income_ratio"]
            ),

            "expense_ratio": float(
                request.POST["expense_ratio"]
            )
        }])

        # =========================================
        # LOAD SVM MODEL
        # =========================================

        svm_model = joblib.load(
            "svm_model.pkl"
        )

        # =========================================
        # LOAD SVM SCALER
        # =========================================

        svm_scaler = joblib.load(
            "svm_scaler.pkl"
        )

        # =========================================
        # SCALE USER DATA
        # SAME SCALER USED DURING SVM TRAINING
        # =========================================

        user_data_scaled = svm_scaler.transform(
            user_data
        )

        # =========================================
        # SVM PREDICTION
        # =========================================

        prediction_result = svm_model.predict(
            user_data_scaled
        )

        # =========================================
        # CONVERT NUMBER TO STATUS
        # =========================================

        predicted_status = target_encoder.inverse_transform(
            prediction_result
        )[0]

        result = (
            "Financial Health Status : "
            + predicted_status
        )

    return render(
        request,
        "prediction.html",
        {
            "result": result
        }
    )

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


        if username and password:

            return redirect("prediction")

        return render(
            request,
            "login.html",
            {
                "error": "Invalid username or password"
            }
        )

    return render(
        request,
        "login.html"
    )


def registration(request):

    error = None
    success = None

    if request.method == "POST":

        # =========================================
        # GET REGISTRATION DATA
        # =========================================

        name = request.POST.get("name")
        age = request.POST.get("age")
        gender = request.POST.get("gender")
        education_level = request.POST.get("education_level")
        employment_status = request.POST.get("employment_status")
        username = request.POST.get("username")
        email = request.POST.get("email")
        phone = request.POST.get("phone")
        password = request.POST.get("password")

        # =========================================
        # VALIDATION
        # =========================================

        if not all([
            name,
            age,
            gender,
            education_level,
            employment_status,
            username,
            email,
            phone,
            password
        ]):
            error = "All fields are required."

            return render(
                request,
                "registration.html",
                {
                    "error": error
                }
            )

        # =========================================
        # USERNAME VALIDATION
        # =========================================

        if len(username) < 6:
            error = "Username must contain at least 6 characters."

            return render(
                request,
                "registration.html",
                {
                    "error": error
                }
            )

        # =========================================
        # PASSWORD VALIDATION
        # =========================================

        if len(password) < 8:
            error = "Password must contain at least 8 characters."

            return render(
                request,
                "registration.html",
                {
                    "error": error
                }
            )

        # =========================================
        # PHONE VALIDATION
        # =========================================

        if not phone.isdigit() or len(phone) != 10:

            error = "Phone number must contain exactly 10 digits."

            return render(
                request,
                "registration.html",
                {
                    "error": error
                }
            )

        # =========================================
        # AGE VALIDATION
        # =========================================

        try:
            age = int(age)

            if age < 18 or age > 100:

                error = "Age must be between 18 and 100."

                return render(
                    request,
                    "registration.html",
                    {
                        "error": error
                    }
                )

        except ValueError:

            error = "Please enter a valid age."

            return render(
                request,
                "registration.html",
                {
                    "error": error
                }
            )

        # =========================================
        # CHECK USERNAME
        # =========================================

        if User.objects.filter(username=username).exists():

            error = "Username already exists."

            return render(
                request,
                "registration.html",
                {
                    "error": error
                }
            )

        # =========================================
        # CHECK EMAIL
        # =========================================

        if User.objects.filter(email=email).exists():

            error = "Email already registered."

            return render(
                request,
                "registration.html",
                {
                    "error": error
                }
            )

        # =========================================
        # CREATE USER
        # =========================================

        user = User.objects.create(
            username=username,
            email=email,
            first_name=name,
            password=make_password(password)
        )

        # =========================================
        # SUCCESS
        # =========================================

        success = "Registration completed successfully."

        return render(
            request,
            "registration.html",
            {
                "success": success
            }
        )

    # =========================================
    # GET REQUEST
    # =========================================

    return render(
        request,
        "registration.html"
    )
# =========================================================
# ASSESSMENT
# =========================================================

def assessment(request):

    financial_status = request.session.get(
        "financial_status"
    )

    credit_score = request.session.get(
        "credit_score"
    )

    return render(
        request,
        "assessment.html",
        {
            "financial_status": financial_status,
            "credit_score": credit_score
        }
    )


# =========================================================
# RECOMMENDATION
# =========================================================

def recommendation(request):

    # =========================================
    # GET DATA FROM SESSION
    # =========================================

    financial_status = request.session.get(
        "financial_status"
    )

    monthly_income = request.session.get(
        "monthly_income",
        0
    )

    monthly_expenses = request.session.get(
        "monthly_expenses",
        0
    )

    savings = request.session.get(
        "savings",
        0
    )

    debt_to_income_ratio = request.session.get(
        "debt_to_income_ratio",
        0
    )

    credit_score = request.session.get(
        "credit_score",
        0
    )


    # =========================================
    # CALCULATE EXPENSE PERCENTAGE
    # =========================================

    if monthly_income > 0:

        expense_percentage = (
            monthly_expenses / monthly_income
        ) * 100

    else:

        expense_percentage = 0


    # =========================================
    # CREATE RECOMMENDATIONS
    # =========================================

    recommendations = []


    # =========================================
    # GOOD FINANCIAL HEALTH
    # =========================================

    if financial_status == "Good":

        recommendations.append({
            "title": "Income Management",
            "message": (
                "Maintain your current spending habits "
                "and continue managing your income wisely."
            )
        })

        recommendations.append({
            "title": "Savings Recommendation",
            "message": (
                "Continue saving regularly and build "
                "a strong emergency fund."
            )
        })

        recommendations.append({
            "title": "Investment Planning",
            "message": (
                "Consider long-term investments to "
                "grow your wealth."
            )
        })

        recommendations.append({
            "title": "Financial Management",
            "message": (
                "Your financial health is good. "
                "Continue maintaining your current habits."
            )
        })


    # =========================================
    # AVERAGE FINANCIAL HEALTH
    # =========================================

    elif financial_status == "Average":

        recommendations.append({
            "title": "Expense Management",
            "message": (
                "Reduce unnecessary expenses and "
                "control your monthly spending."
            )
        })

        recommendations.append({
            "title": "Savings Recommendation",
            "message": (
                "Increase your monthly savings and "
                "maintain an emergency fund."
            )
        })

        recommendations.append({
            "title": "Budget Planning",
            "message": (
                "Create a monthly budget and track "
                "your income and expenses."
            )
        })


    # =========================================
    # POOR FINANCIAL HEALTH
    # =========================================

    elif financial_status == "Poor":

        recommendations.append({
            "title": "Expense Management",
            "message": (
                "Reduce unnecessary expenses immediately "
                "and control your monthly spending."
            )
        })

        recommendations.append({
            "title": "Debt Management",
            "message": (
                "Avoid unnecessary loans and try "
                "to reduce your existing debt."
            )
        })

        recommendations.append({
            "title": "Savings Recommendation",
            "message": (
                "Focus on increasing your savings "
                "and building an emergency fund."
            )
        })

        recommendations.append({
            "title": "Strict Budget Planning",
            "message": (
                "Create and follow a strict monthly budget "
                "to improve your financial condition."
            )
        })


    # =========================================
    # PERSONALIZED EXPENSE RECOMMENDATION
    # =========================================

    if expense_percentage >= 70:

        recommendations.append({
            "title": "High Expense Alert",
            "message": (
                f"Your expenses are {expense_percentage:.2f}% "
                "of your monthly income. Try to reduce "
                "unnecessary spending."
            )
        })


    # =========================================
    # PERSONALIZED DEBT RECOMMENDATION
    # =========================================

    if debt_to_income_ratio > 0.4:

        recommendations.append({
            "title": "Debt Alert",
            "message": (
                "Your debt-to-income ratio is high. "
                "Avoid additional loans and focus "
                "on reducing your debt."
            )
        })


    # =========================================
    # PERSONALIZED CREDIT SCORE RECOMMENDATION
    # =========================================

    if credit_score < 650:

        recommendations.append({
            "title": "Credit Score Improvement",
            "message": (
                "Improve your credit score by making "
                "loan and EMI payments on time."
            )
        })


    return render(
        request,
        "recommendation.html",
        {
            "financial_status": financial_status,
            "monthly_income": monthly_income,
            "monthly_expenses": monthly_expenses,
            "savings": savings,
            "credit_score": credit_score,
            "expense_percentage": expense_percentage,
            "recommendations": recommendations
        }
    )
# =========================================================
# RISK MANAGEMENT
# =========================================================

def riskmanagement(request):

    financial_status = request.session.get(
        "financial_status"
    )


    credit_score = request.session.get(
        "credit_score",
        0
    )

    risks = []

    if financial_status == "Poor":

        risks.append(
            "High financial risk"
        )

        risks.append(
            "Review unnecessary expenses"
        )

        risks.append(
            "Avoid additional debt"
        )

    elif financial_status == "Average":

        risks.append(
            "Moderate financial risk"
        )

        risks.append(
            "Monitor monthly expenses"
        )

    else:

        risks.append(
            "Low financial risk"
        )

        risks.append(
            "Continue monitoring your finances"
        )

    return render(
        request,
        "riskmanagement.html",
        {
            "financial_status": financial_status,
            "credit_score": credit_score,
            "risks": risks
        }
    )

# =========================================================
# CHATBOT
# =========================================================

def chatbot(request):

    response = None

    if request.method == "POST":

        user_message = request.POST.get(
            "message",
            ""
        ).lower()

        # Simple rule-based chatbot initially

        if "saving" in user_message:

            response = (
                "Try to save at least a portion of your "
                "monthly income and maintain an emergency fund."
            )

        elif "expense" in user_message:

            response = (
                "Review unnecessary expenses and create "
                "a monthly spending limit."
            )

        elif "loan" in user_message:

            response = (
                "Avoid unnecessary loans and maintain "
                "your EMI within a manageable level."
            )

        elif "credit" in user_message:

            response = (
                "Your credit score is provided as an input "
                "from your bank or financial statement. "
                "The SVM model uses it as one of the features "
                "for financial health prediction."
            )

        else:

            response = (
                "I can help you with savings, expenses, "
                "loans, credit score and financial planning."
            )

    return render(
        request,
        "chatbot.html",
        {
            "response": response
        }
    )


# =========================================================
# DASHBOARD
# =========================================================

def dashboard(request):

    financial_status = request.session.get(
        "financial_status"
    )

    credit_score = request.session.get(
        "credit_score"
    )

    monthly_income = request.session.get(
        "monthly_income"
    )

    monthly_expenses = request.session.get(
        "monthly_expenses"
    )

    savings = request.session.get(
        "savings"
    )

    return render(
        request,
        "dashboard.html",
        {
            "financial_status": financial_status,
            "credit_score": credit_score,
            "monthly_income": monthly_income,
            "monthly_expenses": monthly_expenses,
            "savings": savings
        }
    )