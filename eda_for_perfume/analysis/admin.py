from django.contrib import admin
from .models import Perfume, PerfumeCategory


@admin.register(PerfumeCategory)
class PerfumeCategoryAdmin(admin.ModelAdmin):
    list_display = ['name', 'gender']
    search_fields = ['name']


@admin.register(Perfume)
class PerfumeAdmin(admin.ModelAdmin):
    list_display = ['name', 'brand', 'category', 'price', 'rating', 'review_count', 'release_year', 'concentration', 'created_at']
    list_filter = ['category', 'concentration', 'release_year']
    search_fields = ['name', 'brand', 'top_notes', 'middle_notes', 'base_notes']
    list_editable = ['price', 'rating', 'review_count']
    ordering = ['-created_at']
    
    fieldsets = (
        ('Basic Info', {
            'fields': ('brand', 'name', 'category')
        }),
        ('Details', {
            'fields': ('top_notes', 'middle_notes', 'base_notes', 'concentration')
        }),
        ('Metrics', {
            'fields': ('price', 'rating', 'review_count', 'release_year')
        }),
    )
