from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator

class Brand(models.Model):
    name = models.CharField(max_length=100, unique=True)
    country = models.CharField(max_length=50, blank=True)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.name

    class Meta:
        ordering = ['name']

class PerfumeCategory(models.Model):
    GENDER_CHOICES = [
        ('Male', 'Nam'),
        ('Female', 'Nữ'),
        ('Unisex', 'Unisex'),
    ]

    name = models.CharField(max_length=50, unique=True)
    gender = models.CharField(max_length=10, choices=GENDER_CHOICES)
    
    def __str__(self):
        return f"{self.name} - {self.gender}"

class Perfume(models.Model):
    # Thông tin cơ bản
    brand = models.ForeignKey(Brand, on_delete=models.CASCADE, related_name='perfumes')
    name = models.CharField(max_length=200)
    category = models.ForeignKey(PerfumeCategory, on_delete=models.SET_NULL, null=True)
    
    # Mùi hương (Notes) - Lưu dạng text, sẽ xử lý ML sau
    top_notes = models.TextField(help_text="Mùi hương đầu, cách nhau bởi dấu phẩy")
    middle_notes = models.TextField(help_text="Mùi hương giữa")
    base_notes = models.TextField(help_text="Mùi hương cuối")
    
    # Thông tin thị trường
    price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    rating = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(5.0)],
        null=True,
        blank=True
    )
    review_count = models.IntegerField(default=0)
    
    # Metadata
    release_year = models.IntegerField(null=True, blank=True)
    concentration = models.CharField(max_length=50, blank=True)  # EDT, EDP, Parfum
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.brand.name} - {self.name}"
    
    class Meta:
        ordering = ['-rating', '-review_count']
        unique_together = ['brand', 'name']

class PredictionHistory(models.Model):
    """Lưu lại các lần dự đoán của user"""
    perfume_name = models.CharField(max_length=200)
    input_notes = models.TextField()
    predicted_rating = models.FloatField()
    actual_rating = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
