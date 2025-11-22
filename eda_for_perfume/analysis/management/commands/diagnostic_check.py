import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from django.core.management.base import BaseCommand
from django.conf import settings

class Command(BaseCommand):
    help = 'Run diagnostic checks on the perfume sales data'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Starting diagnostic checks...'))
        
        # Load and analyze data
        df = self.load_data()
        if df is not None:
            self.analyze_target_variable(df)
            self.analyze_features(df)
            self.assess_data_quality(df)
            self.generate_recommendations(df)
            
        self.stdout.write(self.style.SUCCESS('Diagnostic checks completed!'))

    def load_data(self):
        """Load data from CSV file"""
        try:
            csv_path = os.path.join('analysis', 'ebay_mens_perfume.csv')
            df = pd.read_csv(csv_path)
            self.stdout.write(self.style.SUCCESS(f'✅ Đã load {len(df)} dòng dữ liệu'))
            return df
        except Exception as e:
            self.stderr.write(self.style.ERROR(f'❌ Lỗi khi load dữ liệu: {e}'))
            return None

    def analyze_target_variable(self, df, target_col='sold'):
        """Analyze the target variable"""
        self.stdout.write("\n" + "="*80)
        self.stdout.write("📊 PHÂN TÍCH BIẾN MỤC TIÊU (SOLD)")
        self.stdout.write("="*80)
        
        if target_col not in df.columns:
            self.stderr.write(f"❌ Không tìm thấy cột '{target_col}' trong dữ liệu")
            return
        
        # Basic statistics
        sold = df[target_col]
        self.stdout.write("\n📈 Thống kê cơ bản:")
        self.stdout.write(str(sold.describe()))
        
        # Save distribution plot
        self.save_plot(
            lambda: plt.hist(sold, bins=30),
            'Phân bố số lượng bán hàng',
            'Số lượng đã bán',
            'Tần suất',
            'sold_distribution.png'
        )

    def analyze_features(self, df, target_col='sold'):
        """Analyze features in the dataset"""
        self.stdout.write("\n" + "="*80)
        self.stdout.write("🔍 PHÂN TÍCH CÁC ĐẶC TRƯNG")
        self.stdout.write("="*80)
        
        # Check for missing values
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        missing_df = pd.DataFrame({'missing_count': missing, 'missing_percentage': missing_pct})
        missing_df = missing_df[missing_df['missing_count'] > 0].sort_values('missing_count', ascending=False)
        
        if not missing_df.empty:
            self.stdout.write("\n❌ Các cột bị thiếu dữ liệu:")
            self.stdout.write(missing_df.to_string())
        else:
            self.stdout.write("\n✅ Không có dữ liệu bị thiếu")
        
        # Analyze numerical features
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        if numeric_cols:
            self.stdout.write("\n📈 Thống kê các đặc trưng số:")
            self.stdout.write(df[numeric_cols].describe().T.to_string())
            
            # Correlation with target
            if target_col in df.columns:
                corr = df[numeric_cols + [target_col]].corr()[target_col].sort_values(ascending=False)
                self.stdout.write("\n📊 Tương quan với biến mục tiêu (sold):")
                self.stdout.write(corr.to_string())
                
                # Plot correlation heatmap
                plt.figure(figsize=(10, 8))
                sns.heatmap(df[numeric_cols + [target_col]].corr(), annot=True, cmap='coolwarm', center=0)
                self.save_plot(
                    None,  # We already created the plot
                    'Ma trận tương quan',
                    '',
                    '',
                    'correlation_heatmap.png',
                    save_fig=False
                )
    
    def assess_data_quality(self, df, target_col='sold'):
        """Assess overall data quality"""
        self.stdout.write("\n" + "="*80)
        self.stdout.write("🏆 ĐÁNH GIÁ CHẤT LƯỢNG DỮ LIỆU")
        self.stdout.write("="*80)
        
        issues = []
        
        # Check sample size
        if len(df) < 500:
            issues.append("⚠️  Kích thước mẫu nhỏ ({} dòng), có thể không đủ để huấn luyện mô hình hiệu quả".format(len(df)))
        
        # Check target variable
        if target_col in df.columns:
            # Check for class imbalance (for classification)
            if df[target_col].nunique() < 10:  # Assuming classification if few unique values
                class_dist = df[target_col].value_counts(normalize=True)
                if (class_dist < 0.1).any():
                    issues.append("❌ Mất cân bằng lớp nghiêm trọng: {}".format(class_dist.to_dict()))
            
            # Check for zero-inflation (for regression)
            if df[target_col].nunique() > 10:  # Assuming regression if many unique values
                zero_count = (df[target_col] == 0).sum()
                if zero_count / len(df) > 0.3:
                    issues.append("❌ Quá nhiều giá trị 0 trong biến mục tiêu ({:.1f}%)".format(zero_count/len(df)*100))
        
        # Check missing values
        missing_cols = df.isnull().sum()
        missing_cols = missing_cols[missing_cols > 0]
        if not missing_cols.empty:
            issues.append("❌ Có {} cột chứa giá trị thiếu".format(len(missing_cols)))
        
        # Check constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() == 1]
        if constant_cols:
            issues.append("❌ Các cột hằng số (không có thông tin): {}".format(constant_cols))
        
        # Check duplicate rows
        if df.duplicated().sum() > 0:
            issues.append("❌ Phát hiện {} dòng trùng lặp".format(df.duplicated().sum()))
        
        # Print results
        if issues:
            self.stdout.write("\n🚨 VẤN ĐỀ CẦN XỬ LÝ:")
            for issue in issues:
                self.stdout.write("- " + issue)
        else:
            self.stdout.write("\n✅ Không phát hiện vấn đề nghiêm trọng")

    def generate_recommendations(self, df, target_col='sold'):
        """Generate recommendations for data improvement"""
        self.stdout.write("\n" + "="*80)
        self.stdout.write("💡 ĐỀ XUẤT CẢI THIỆN")
        self.stdout.write("="*80)
        
        self.stdout.write("\n1️⃣  TIỀN XỬ LÝ DỮ LIỆU:")
        self.stdout.write("  - Xử lý giá trị thiếu:")
        self.stdout.write("    * Sử dụng giá trị trung bình/trung vị cho biến số")
        self.stdout.write("    * Sử dụng giá trị phổ biến nhất cho biến phân loại")
        
        self.stdout.write("\n  - Xử lý ngoại lai:")
        self.stdout.write("    * Phát hiện và xử lý các giá trị ngoại lai bằng IQR hoặc Z-score")
        
        if target_col in df.columns and df[target_col].nunique() > 10:  # If regression
            self.stdout.write("\n2️⃣  BIẾN ĐỔI DỮ LIỆU:")
            self.stdout.write("  - Áp dụng log transformation cho biến mục tiêu nếu bị lệch phải")
            
        self.stdout.write("\n3️⃣  KỸ THUẬT MÃ HÓA:")
        self.stdout.write("  - Mã hóa one-hot cho các biến phân loại có ít giá trị duy nhất")
        self.stdout.write("  - Sử dụng target encoding cho các biến phân loại có nhiều giá trị duy nhất")

    def save_plot(self, plot_func, title, xlabel, ylabel, filename, save_fig=True):
        """Helper function to save plots"""
        try:
            plt.figure(figsize=(10, 6))
            if plot_func is not None:
                plot_func()
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.grid(True)
            
            # Ensure the directory exists
            os.makedirs(os.path.join('static', 'analysis', 'plots'), exist_ok=True)
            plot_path = os.path.join('static', 'analysis', 'plots', filename)
            plt.savefig(plot_path)
            plt.close()
            self.stdout.write(f"💾 Đã lưu biểu đồ: {plot_path}")
        except Exception as e:
            self.stderr.write(f"❌ Lỗi khi lưu biểu đồ: {e}")
