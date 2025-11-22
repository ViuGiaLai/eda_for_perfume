from django.core.management.base import BaseCommand
from analysis.models import Brand, PerfumeCategory, Perfume
from analysis.ml_models.data_loader import PerfumeDataLoader
import pandas as pd

class Command(BaseCommand):
    help = 'Khởi tạo dữ liệu mẫu cho database'
    
    def handle(self, *args, **kwargs):
        self.stdout.write('🚀 Bắt đầu khởi tạo dữ liệu...')
        
        # 1. Load data
        loader = PerfumeDataLoader()
        df = loader.load_data()
        df = loader.clean_data()
        
        # 2. Tạo Brands
        self.stdout.write('→ Tạo Brands...')
        brands = {}
        for brand_name in df['Brand'].unique():
            brand, created = Brand.objects.get_or_create(
                name=brand_name,
                defaults={'country': 'France'}  # Mặc định
            )
            brands[brand_name] = brand
        
        # 3. Tạo Categories
        self.stdout.write('→ Tạo Categories...')
        categories = {}
        for gender in df['Gender'].unique():
            cat, created = PerfumeCategory.objects.get_or_create(
                name=f'{gender} Fragrance',
                gender=gender
            )
            categories[gender] = cat
        
        # 4. Tạo Perfumes
        self.stdout.write('→ Tạo Perfumes...')
        created_count = 0
        
        for idx, row in df.iterrows():
            # Xử lý concentration - đảm bảo giá trị hợp lệ
            concentration = row.get('Concentration', 'EDP')
            if pd.isna(concentration) or concentration == '/':
                concentration = 'EDP'
            
            # Xử lý các giá trị có thể là NaN
            review_count = row['Review_Count']
            if pd.isna(review_count):
                review_count = 0
            else:
                review_count = int(review_count)
            
            perfume, created = Perfume.objects.get_or_create(
                brand=brands[row['Brand']],
                name=row['Name'],
                defaults={
                    'category': categories[row['Gender']],
                    'top_notes': row['Top_Notes'],
                    'middle_notes': row['Middle_Notes'],
                    'base_notes': row['Base_Notes'],
                    'price': float(row['Price']),
                    'rating': float(row['Rating']),
                    'review_count': review_count,
                    'release_year': int(row['Release_Year']),
                    'concentration': concentration
                }
            )
            if created:
                created_count += 1
        
        self.stdout.write(
            self.style.SUCCESS(
                f'✅ Hoàn thành! Đã tạo {created_count} nước hoa mới'
            )
        )
