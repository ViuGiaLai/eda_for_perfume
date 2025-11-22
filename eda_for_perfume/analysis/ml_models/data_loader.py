import pandas as pd
import numpy as np
from pathlib import Path

class ImprovedPerfumeDataLoader:
    """Data Loader cải tiến với xử lý outliers và feature engineering tốt hơn"""
    
    def __init__(self, csv_path='analysis/ebay_mens_perfume.csv'):
        self.csv_path = csv_path
        self.df = None
        self.outlier_removed_count = 0
        
    def load_data(self):
        """Load dữ liệu từ CSV"""
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"✅ Đã load {len(self.df)} dòng dữ liệu từ eBay")
            return self.df
        except FileNotFoundError:
            print("⚠️  Không tìm thấy file CSV, tạo dữ liệu mẫu...")
            self.df = self._create_sample_data()
            return self.df
    
    def _create_sample_data(self):
        """Tạo dữ liệu mẫu"""
        data = {
            'brand': ['Dior', 'Chanel', 'Gucci', 'Tom Ford', 'Versace'] * 40,
            'title': ['Sauvage EDT', 'Bleu de Chanel', 'Guilty Pour Homme', 
                     'Oud Wood', 'Eros EDT'] * 40,
            'type': np.random.choice(['EDT', 'EDP', 'Parfum', 'Cologne'], 200),
            'price': np.random.uniform(30, 300, 200),
            'priceWithCurrency': ['$' + str(round(p, 2)) for p in np.random.uniform(30, 300, 200)],
            'available': np.random.randint(0, 100, 200),
            'availableText': [f"{a} available" for a in np.random.randint(0, 100, 200)],
            'sold': np.random.randint(0, 500, 200),
            'lastUpdated': pd.date_range(start='2023-01-01', periods=200, freq='D'),
            'itemLocation': np.random.choice(['New York, US', 'London, UK', 'Paris, FR', 
                                            'Berlin, DE', 'Tokyo, JP'], 200)
        }
        return pd.DataFrame(data)
    
    def remove_outliers_iqr(self, column='Sold', multiplier=1.5):
        """
        ✅ XỬ LÝ OUTLIERS BẰNG IQR METHOD
        Đây là bước QUAN TRỌNG NHẤT để cải thiện mô hình
        """
        if column not in self.df.columns:
            print(f"⚠️  Cột {column} không tồn tại")
            return
            
        before_count = len(self.df)
        
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        # Lọc outliers
        self.df = self.df[
            (self.df[column] >= lower_bound) & 
            (self.df[column] <= upper_bound)
        ].copy()
        
        self.outlier_removed_count = before_count - len(self.df)
        
        print(f"\n🔧 XỬ LÝ OUTLIERS ({column}):")
        print(f"   Q1 = {Q1:.1f}, Q3 = {Q3:.1f}, IQR = {IQR:.1f}")
        print(f"   Lower bound = {lower_bound:.1f}")
        print(f"   Upper bound = {upper_bound:.1f}")
        print(f"   ❌ Đã loại bỏ {self.outlier_removed_count} outliers ({self.outlier_removed_count/before_count*100:.1f}%)")
        print(f"   ✅ Còn lại {len(self.df)} dòng sạch")
    
    def apply_log_transform(self, column='Sold'):
        """
        ✅ LOG TRANSFORM CHO DỮ LIỆU LỆCH PHẢI
        Biến đổi này giúp chuẩn hóa phân phối và cải thiện mô hình
        """
        if column not in self.df.columns:
            return
            
        # Lưu cột gốc
        self.df[f'{column}_Original'] = self.df[column].copy()
        
        # Apply log1p (log(1+x) để xử lý giá trị 0)
        self.df[f'{column}_Log'] = np.log1p(self.df[column])
        
        print(f"\n📊 LOG TRANSFORM ({column}):")
        print(f"   Trước transform: mean={self.df[column].mean():.1f}, std={self.df[column].std():.1f}")
        print(f"   Sau transform:  mean={self.df[f'{column}_Log'].mean():.3f}, std={self.df[f'{column}_Log'].std():.3f}")
        print(f"   ✅ Đã tạo cột mới: {column}_Log")
    
    def clean_data(self, remove_outliers=True, log_transform=True):
        """
        ✅ LÀM SẠCH DỮ LIỆU VỚI CÁC KỸ THUẬT NÂNG CAO
        """
        print("\n" + "="*80)
        print("🧹 BẮT ĐẦU LÀM SẠCH DỮ LIỆU (IMPROVED VERSION)")
        print("="*80)
        
        original_count = len(self.df)
        
        # 1. Đổi tên cột
        if 'brand' in self.df.columns:
            self.df = self.df.rename(columns={
                'brand': 'Brand',
                'title': 'Title',
                'type': 'Type',
                'price': 'Price',
                'sold': 'Sold'
            })
        
        # 2. Xử lý giá trị thiếu
        print("\n📝 Xử lý giá trị thiếu...")
        
        # Drop rows thiếu dữ liệu quan trọng
        critical_cols = ['Brand', 'Title', 'Sold']
        before_drop = len(self.df)
        self.df = self.df.dropna(subset=critical_cols)
        print(f"   ❌ Đã xóa {before_drop - len(self.df)} dòng thiếu dữ liệu quan trọng")
        
        # Điền giá trị thiếu cho numerical features
        if 'Price' in self.df.columns:
            self.df['Price'] = pd.to_numeric(self.df['Price'], errors='coerce')
            median_price = self.df['Price'].median()
            self.df['Price'] = self.df['Price'].fillna(median_price)
            print(f"   ✅ Điền Price thiếu bằng median: {median_price:.2f}")
        
        if 'available' in self.df.columns:
            self.df['Available'] = pd.to_numeric(self.df['available'], errors='coerce')
            median_available = self.df['Available'].median()
            self.df['Available'] = self.df['Available'].fillna(median_available).astype(int)
            print(f"   ✅ Điền Available thiếu bằng median: {median_available:.0f}")
        else:
            self.df['Available'] = 0
        
        # Điền categorical features bằng mode
        if 'Type' in self.df.columns:
            mode_type = self.df['Type'].mode()[0] if len(self.df['Type'].mode()) > 0 else 'EDT'
            self.df['Type'] = self.df['Type'].fillna(mode_type)
            print(f"   ✅ Điền Type thiếu bằng mode: {mode_type}")
        
        # 3. Xử lý Target Variable (Sold)
        print("\n🎯 Xử lý Target Variable (Sold)...")
        self.df['Sold'] = pd.to_numeric(self.df['Sold'], errors='coerce')
        self.df['Sold'] = self.df['Sold'].fillna(0).astype(int)
        
        print(f"   Trước xử lý: min={self.df['Sold'].min()}, max={self.df['Sold'].max()}, mean={self.df['Sold'].mean():.1f}, std={self.df['Sold'].std():.1f}")
        
        # ✅ XỬ LÝ OUTLIERS (Quan trọng nhất!)
        if remove_outliers:
            self.remove_outliers_iqr('Sold', multiplier=1.5)
        
        # ✅ LOG TRANSFORM (Chuẩn hóa phân phối)
        if log_transform:
            self.apply_log_transform('Sold')
        
        # 4. Xử lý các features khác
        self.df['Is_Available'] = (self.df['Available'] > 0).astype(int)
        
        # lastUpdated
        if 'lastUpdated' in self.df.columns:
            self.df['lastUpdated'] = pd.to_datetime(self.df['lastUpdated'], errors='coerce')
            now = pd.Timestamp.now()
            self.df['Days_Since_Update'] = (now - self.df['lastUpdated']).dt.total_seconds() / 86400
            median_days = self.df['Days_Since_Update'].median()
            self.df['Days_Since_Update'] = self.df['Days_Since_Update'].fillna(median_days).astype(float)
        else:
            self.df['Days_Since_Update'] = 0.0
        
        # itemLocation
        if 'itemLocation' in self.df.columns:
            parts = self.df['itemLocation'].astype(str).str.split(',')
            self.df['Country'] = parts.apply(lambda x: x[-1].strip() if len(x) >= 1 else 'Unknown')
            self.df['State_City'] = parts.apply(lambda x: x[0].strip() if len(x) >= 1 else 'Unknown')
        else:
            self.df['Country'] = 'Unknown'
            self.df['State_City'] = 'Unknown'
        
        # 5. ✅ FEATURE ENGINEERING NÂNG CAO
        print("\n🔧 Feature Engineering nâng cao...")
        
        # Title features
        self.df['Title_Length'] = self.df['Title'].str.len()
        self.df['Title_Word_Count'] = self.df['Title'].str.split().str.len()
        
        # Price features
        self.df['Price_Bucket'] = pd.qcut(self.df['Price'], q=5, labels=False, duplicates='drop')
        self.df['Is_High_Price'] = (self.df['Price'] > self.df['Price'].median()).astype(int)
        self.df['Is_Low_Price'] = (self.df['Price'] < self.df['Price'].quantile(0.25)).astype(int)
        self.df['Log_Price'] = np.log1p(self.df['Price'])
        
        # Available features
        self.df['Available_Bucket'] = pd.qcut(self.df['Available'], q=4, labels=False, duplicates='drop')
        self.df['Log_Available'] = np.log1p(self.df['Available'])
        
        # Interaction features
        self.df['Price_Per_Available'] = self.df['Price'] / (self.df['Available'] + 1)
        self.df['Price_Times_Available'] = self.df['Price'] * self.df['Available']
        
        # Time features
        self.df['Is_Recently_Updated'] = (self.df['Days_Since_Update'] < 7).astype(int)
        self.df['Is_Very_Old'] = (self.df['Days_Since_Update'] > 90).astype(int)
        
        print(f"   ✅ Đã tạo 14+ features mới")
        
        # 6. Xóa duplicates
        before = len(self.df)
        self.df = self.df.drop_duplicates(subset=['Brand', 'Title', 'Price'])
        print(f"\n🔄 Đã xóa {before - len(self.df)} dòng trùng lặp")
        
        # Summary
        print("\n" + "="*80)
        print("✅ HOÀN THÀNH LÀM SẠCH DỮ LIỆU")
        print("="*80)
        print(f"📊 Kết quả:")
        print(f"   • Dữ liệu ban đầu: {original_count} dòng")
        print(f"   • Sau xử lý: {len(self.df)} dòng ({len(self.df)/original_count*100:.1f}%)")
        print(f"   • Target (Sold): min={self.df['Sold'].min()}, max={self.df['Sold'].max()}, mean={self.df['Sold'].mean():.1f}")
        
        if 'Sold_Log' in self.df.columns:
            print(f"   • Target (Sold_Log): mean={self.df['Sold_Log'].mean():.3f}, std={self.df['Sold_Log'].std():.3f}")
        
        return self.df
    
    def get_statistics(self):
        """Thống kê mô tả chi tiết"""
        print("\n" + "="*80)
        print("📊 THỐNG KÊ MÔ TẢ CHI TIẾT")
        print("="*80)
        
        # Numerical features
        num_cols = ['Price', 'Sold', 'Available', 'Days_Since_Update']
        num_cols = [c for c in num_cols if c in self.df.columns]
        
        if num_cols:
            print("\n📈 Đặc trưng số:")
            print(self.df[num_cols].describe())
        
        # Categorical features
        print("\n🏷️  TOP 10 BRANDS THEO DOANH SỐ:")
        if 'Brand' in self.df.columns and 'Sold' in self.df.columns:
            top_brands = self.df.groupby('Brand')['Sold'].sum().sort_values(ascending=False).head(10)
            for brand, sold in top_brands.items():
                print(f"   {brand}: {sold:.0f} sản phẩm")
        
        print("\n🌍 PHÂN BỐ THEO QUỐC GIA:")
        if 'Country' in self.df.columns:
            country_dist = self.df['Country'].value_counts().head(10)
            for country, count in country_dist.items():
                print(f"   {country}: {count} sản phẩm")
        
        # Correlations
        print("\n🔗 TƯƠNG QUAN VỚI TARGET (SOLD):")
        if 'Sold' in self.df.columns:
            corr_cols = ['Price', 'Available', 'Days_Since_Update', 'Title_Length', 
                        'Log_Price', 'Log_Available', 'Price_Per_Available']
            corr_cols = [c for c in corr_cols if c in self.df.columns]
            
            if corr_cols:
                correlations = self.df[corr_cols + ['Sold']].corr()['Sold'].sort_values(ascending=False)
                for col, corr_val in correlations.items():
                    if col != 'Sold':
                        indicator = "🔴" if abs(corr_val) < 0.1 else "🟡" if abs(corr_val) < 0.3 else "🟢"
                        print(f"   {indicator} {col}: {corr_val:.3f}")
        
        print("\n" + "="*80)