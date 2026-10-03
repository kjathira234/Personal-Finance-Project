from django.db import models
from django.contrib.auth.models import User


class Income(models.Model):

    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='incomes'
    )

    source = models.CharField(
        max_length=100
    )

    amount = models.DecimalField(
        max_digits=12,
        decimal_places=2
    )

    date = models.DateField()

    description = models.TextField(
        blank=True,
        null=True
    )

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    def __str__(self):
        return f"{self.source} - ₹{self.amount}"


class Expense(models.Model):

    CATEGORY_CHOICES = [
        ('Grocery', 'Grocery'),
        ('Personal Care', 'Personal Care'),
        ('Maintenance', 'Maintenance'),
        ('Transportation', 'Transportation'),
        ('Education', 'Education'),
        ('Medical', 'Medical'),
        ('Entertainment', 'Entertainment'),
        ('Shopping', 'Shopping'),
        ('Rent', 'Rent'),
        ('Other', 'Other'),
    ]

    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='expenses'
    )

    category = models.CharField(
        max_length=50,
        choices=CATEGORY_CHOICES
    )

    amount = models.DecimalField(
        max_digits=12,
        decimal_places=2
    )

    date = models.DateField()

    description = models.TextField(
        blank=True,
        null=True
    )

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    def __str__(self):
        return f"{self.category} - ₹{self.amount}"


class Loan(models.Model):

    LOAN_TYPE_CHOICES = [
        ('Education', 'Education Loan'),
        ('Housing', 'Housing Loan'),
        ('Vehicle', 'Vehicle Loan'),
        ('Personal', 'Personal Loan'),
        ('Other', 'Other'),
    ]

    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='loans'
    )

    loan_type = models.CharField(
        max_length=50,
        choices=LOAN_TYPE_CHOICES
    )

    amount = models.DecimalField(
        max_digits=12,
        decimal_places=2
    )

    monthly_emi = models.DecimalField(
        max_digits=12,
        decimal_places=2
    )

    date = models.DateField()

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    def __str__(self):
        return f"{self.loan_type} - ₹{self.amount}"

class UserProfile(models.Model):

    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        related_name='profile'
    )

    age = models.IntegerField()

    gender = models.CharField(
        max_length=20
    )

    education_level = models.CharField(
        max_length=50
    )

    employment_status = models.CharField(
        max_length=50
    )

    def __str__(self):
        return f"{self.user.username} Profile"


class Income(models.Model):

    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='incomes'
    )

    source = models.CharField(
        max_length=100
    )

    amount = models.DecimalField(
        max_digits=12,
        decimal_places=2
    )

    date = models.DateField()

    description = models.TextField(
        blank=True,
        null=True
    )

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    def __str__(self):
        return f"{self.source} - ₹{self.amount}"