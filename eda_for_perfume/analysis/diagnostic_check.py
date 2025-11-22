"""
Script kiểm tra chất lượng dữ liệu và đề xuất cải thiện
Chạy: python manage.py shell < diagnostic_check.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# Set up paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = os.path.join(BASE_DIR, 'ebay_mens_perfume.csv')

def load_data():
    """Load data from CSV file"""
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"✅ Đã load {len(df)} dòng dữ liệu từ {DATA_PATH}")
        return df
    except Exception as e:
        print(f"❌ Lỗi khi load dữ liệu: {e}")
        return None

def analyze_target_variable(df, target_col='sold'):
    """Phân tích biến mục tiêu (sold)"""
    print("\n" + "="*80)
    print("📊 PHÂN TÍCH BIẾN MỤC TIÊU (SOLD)")
    print("="*80)
    
    if target_col not in df.columns:
        print(f"❌ Không tìm thấy cột '{target_col}' trong dữ liệu")
        return
    
    # Basic statistics
    sold = df[target_col]
    print(f"\n📈 Thống kê cơ bản:")
    print(sold.describe())
    
    # Missing values
    missing = sold.isnull().sum()
    print(f"\n🔍 Giá trị thiếu: {missing} ({missing/len(sold)*100:.1f}%)")
    
    # Zero values
    zero_count = (sold == 0).sum()
    print(f"\n🔢 Số lượng giá trị 0: {zero_count} ({zero_count/len(sold)*100:.1f}%)")
    
    # Distribution analysis
    print("\n📊 Phân vị:")
    percentiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    for p in percentiles:
        val = sold.quantile(p)
        print(f"  {int(p*100)}%: {val:.1f}")
    
    # Outliers detection
    q1 = sold.quantile(0.25)
    q3 = sold.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = df[(sold < lower_bound) | (sold > upper_bound)]
    print(f"\n⚠️  Phát hiện {len(outliers)} outliers (theo phương pháp IQR)")
    
    # Distribution plot
    plt.figure(figsize=(12, 6))
    sns.histplot(sold, kde=True, bins=50)
    plt.title('Phân bố của biến mục tiêu (sold)')
    plt.xlabel('Số lượng đã bán')
    plt.ylabel('Tần suất')
    plt.grid(True)
    
    # Save plot
    plot_path = os.path.join(BASE_DIR, 'static', 'analysis', 'sold_distribution.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    print(f"\n💾 Đã lưu biểu đồ phân bố tại: {plot_path}")
    plt.close()

def analyze_features(df, target_col='sold'):
    """Phân tích các đặc trưng"""
    print("\n" + "="*80)
    print("🔍 PHÂN TÍCH CÁC ĐẶC TRƯNG")
    print("="*80)
    
    # Check for missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({'missing_count': missing, 'missing_percentage': missing_pct})
    missing_df = missing_df[missing_df['missing_count'] > 0].sort_values('missing_count', ascending=False)
    
    if not missing_df.empty:
        print("\n❌ Các cột bị thiếu dữ liệu:")
        print(missing_df)
    else:
        print("\n✅ Không có dữ liệu bị thiếu")
    
    # Analyze numerical features
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    if numeric_cols:
        print("\n📈 Thống kê các đặc trưng số:")
        print(df[numeric_cols].describe().T)
        # Correlation with target
        if target_col in df.columns:
            corr = df[numeric_cols + [target_col]].corr()[target_col].sort_values(ascending=False)
            print("\n📊 Tương quan với biến mục tiêu (sold):")
            print(corr)
            
            # Plot correlation heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(df[numeric_cols + [target_col]].corr(), annot=True, cmap='coolwarm', center=0)
            plt.title('Ma trận tương quan')
            
            # Save correlation plot
            corr_plot_path = os.path.join(BASE_DIR, 'static', 'analysis', 'correlation_heatmap.png')
            plt.savefig(corr_plot_path)
            print(f"\n💾 Đã lưu biểu đồ tương quan tại: {corr_plot_path}")
            plt.close()
    
    # Analyze categorical features
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        print("\n📊 Thống kê các đặc trưng phân loại:")
        for col in cat_cols:
            print(f"\n🔤 {col}:")
            print(f"Số lượng giá trị duy nhất: {df[col].nunique()}")
            print("Giá trị phổ biến:")
            print(df[col].value_counts().head())

def data_quality_assessment(df, target_col='sold'):
    """Đánh giá tổng quan chất lượng dữ liệu"""
    print("\n" + "="*80)
    print("🏆 ĐÁNH GIÁ CHẤT LƯỢNG DỮ LIỆU")
    print("="*80)
    
    issues = []
    warnings = []
    
    # 1. Check sample size
    if len(df) < 500:
        warnings.append(f"⚠️  Kích thước mẫu nhỏ ({len(df)} dòng), có thể không đủ để huấn luyện mô hình hiệu quả")
    
    # 2. Check target variable
    if target_col in df.columns:
        # Check for class imbalance (for classification)
        if df[target_col].nunique() < 10:  # Assuming classification if few unique values
            class_dist = df[target_col].value_counts(normalize=True)
            if (class_dist < 0.1).any():
                issues.append(f"❌ Mất cân bằng lớp nghiêm trọng: {class_dist.to_dict()}")
        
        # Check for zero-inflation (for regression)
        if df[target_col].nunique() > 10:  # Assuming regression if many unique values
            zero_count = (df[target_col] == 0).sum()
            if zero_count / len(df) > 0.3:
                issues.append(f"❌ Quá nhiều giá trị 0 trong biến mục tiêu ({zero_count/len(df)*100:.1f}%)")
    
    # 3. Check missing values
    missing_cols = df.isnull().sum()
    missing_cols = missing_cols[missing_cols > 0]
    if not missing_cols.empty:
        issues.append(f"❌ Có {len(missing_cols)} cột chứa giá trị thiếu")
    
    # 4. Check constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    if constant_cols:
        issues.append(f"❌ Các cột hằng số (không có thông tin): {constant_cols}")
    
    # 5. Check duplicate rows
    if df.duplicated().sum() > 0:
        issues.append(f"❌ Phát hiện {df.duplicated().sum()} dòng trùng lặp")
    
    # Print results
    if issues:
        print("\n🚨 VẤN ĐỀ CẦN XỬ LÝ:")
        for issue in issues:
            print(f"- {issue}")
    else:
        print("\n✅ Không phát hiện vấn đề nghiêm trọng")
    
    if warnings:
        print("\n⚠️  CẢNH BÁO:")
        for warning in warnings:
            print(f"- {warning}")

def generate_recommendations(df, target_col='sold'):
    """Tạo các khuyến nghị cải thiện dữ liệu"""
    print("\n" + "="*80)
    print("💡 ĐỀ XUẤT CẢI THIỆN")
    print("="*80)
    
    print("\n1️⃣  TIỀN XỬ LÝ DỮ LIỆU:")
    print("  - Xử lý giá trị thiếu:")
    print("    * Sử dụng giá trị trung bình/trung vị cho biến số")
    print("    * Sử dụng giá trị phổ biến nhất cho biến phân loại")
    print("    * Hoặc xóa các dòng chứa giá trị thiếu nếu ít")
    
    print("\n  - Xử lý ngoại lai:")
    print("    * Phát hiện và xử lý các giá trị ngoại lai bằng IQR hoặc Z-score")
    print("    * Cân nhắc sử dụng log transformation cho biến lệch phải")
    
    print("\n2️⃣  KỸ THUẬT LẤY MẪU:")
    if (df[target_col] == 0).mean() > 0.3:
        print("  - Cân nhắc sử dụng kỹ thuật lấy mẫu lại (resampling):")
        print("    * Oversampling cho lớp thiểu số")
        print("    * Undersampling cho lớp đa số")
        print("    * SMOTE để tạo mẫu tổng hợp")
    
    print("\n3️⃣  KỸ THUẬT MÃ HÓA:")
    print("  - Mã hóa one-hot cho các biến phân loại có ít giá trị duy nhất")
    print("  - Sử dụng target encoding cho các biến phân loại có nhiều giá trị duy nhất")
    print("  - Chuẩn hóa (scale) các đặc trưng số về cùng một khoảng giá trị")
    
    print("\n4️⃣  KỸ THUẬT KHAI PHÁ ĐẶC TRƯNG:")
    print("  - Tạo các đặc trưng tương tác giữa các biến")
    print("  - Trích xuất thông tin từ văn bản (nếu có)")
    print("  - Tạo các đặc trưng thống kê theo nhóm")
    
    print("\n5️⃣  MÔ HÌNH DỰ ĐOÁN:")
    print("  - Thử nghiệm nhiều mô hình khác nhau: XGBoost, LightGBM, Random Forest")
    print("  - Sử dụng cross-validation để đánh giá hiệu suất ổn định")
    print("  - Tối ưu hyperparameters bằng GridSearch hoặc Bayesian Optimization")

def main():
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Basic info
    print("\n📋 THÔNG TIN CƠ BẢN VỀ DỮ LIỆU")
    print("="*50)
    print(f"Tổng số dòng: {len(df)}")
    print(f"Tổng số cột: {len(df.columns)}")
    print("\nCác cột trong dữ liệu:", ", ".join(df.columns.tolist()))
    
    # Perform analysis
    analyze_target_variable(df)
    analyze_features(df)
    data_quality_assessment(df)
    generate_recommendations(df)
    
    print("\n✅ Hoàn thành phân tích dữ liệu!")

if __name__ == "__main__":
    main()
