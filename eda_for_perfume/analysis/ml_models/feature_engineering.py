from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
import pandas as pd
import numpy as np

class ImprovedFeatureEngineer:
    """
    ✅ Feature Engineering cải tiến với:
    - RobustScaler (tốt hơn StandardScaler cho dữ liệu có outliers)
    - Target Encoding cho categorical features nhiều giá trị
    - Polynomial features cho interaction
    - Xử lý unseen labels tốt hơn
    """
    
    def __init__(self, use_log_target=True):
        self.title_vectorizer = None
        self.label_encoders = {}
        self.target_encoders = {}  # NEW: Target encoding
        self.scaler = RobustScaler()  # ✅ Thay StandardScaler → RobustScaler
        self.feature_names = []
        self.use_log_target = use_log_target
        self.target_mean = None
        
    def engineer_features(self, df):
        """
        ✅ FEATURE ENGINEERING CẢI TIẾN
        Tạo features phong phú hơn để mô hình học tốt hơn
        """
        print("\n" + "="*80)
        print("🔧 BẮT ĐẦU FEATURE ENGINEERING (IMPROVED VERSION)")
        print("="*80)
        
        df = df.copy()
        
        # ========================================
        # 1. TEXT FEATURES (TF-IDF từ Title)
        # ========================================
        print("\n📝 Xử lý Text Features (TF-IDF)...")
        self.title_vectorizer = TfidfVectorizer(
            max_features=60,  # Giảm bớt số lượng feature để tránh overfit
            stop_words='english',
            ngram_range=(1, 2),  # Chỉ dùng unigram + bigram cho ổn định hơn
            min_df=2,  # Bỏ từ xuất hiện quá ít
            max_df=0.8  # Bỏ từ xuất hiện quá nhiều
        )
        
        title_matrix = self.title_vectorizer.fit_transform(df['Title'].fillna(''))
        title_df = pd.DataFrame(
            title_matrix.toarray(),
            columns=[f"title_{name}" for name in self.title_vectorizer.get_feature_names_out()]
        )
        print(f"   ✅ Tạo {len(title_df.columns)} TF-IDF features từ Title")
        
        # ========================================
        # 2. CATEGORICAL ENCODING (Label + Target)
        # ========================================
        print("\n🏷️  Encoding Categorical Variables...")
        categorical_cols = ['Brand', 'Type', 'Country', 'State_City']
        
        for col in categorical_cols:
            if col not in df.columns:
                continue
                
            # Label Encoding
            le = LabelEncoder()
            series = df[col].astype(str).fillna('Unknown')
            
            # Đảm bảo 'Unknown' có trong fit
            if 'Unknown' not in series.unique():
                series = pd.concat([series, pd.Series(['Unknown'])], ignore_index=True)
            
            le.fit(series)
            df[f'{col}_encoded'] = le.transform(df[col].astype(str).fillna('Unknown'))
            self.label_encoders[col] = le
            
            # ✅ TARGET ENCODING (mạnh hơn Label Encoding)
            # Chỉ áp dụng nếu có nhiều unique values
            if df[col].nunique() > 5:
                target_mean = df.groupby(col)['Sold'].mean()
                self.target_encoders[col] = target_mean
                df[f'{col}_target_encoded'] = df[col].map(target_mean).fillna(df['Sold'].mean())
                print(f"   ✅ {col}: Label + Target Encoding (unique={df[col].nunique()})")
            else:
                print(f"   ✅ {col}: Label Encoding only (unique={df[col].nunique()})")
        
        # ========================================
        # 3. NUMERICAL FEATURES (Existing + New)
        # ========================================
        print("\n🔢 Xử lý Numerical Features...")
        
        numerical_features = []
        
        # Basic numerical
        basic_nums = ['Price', 'Available', 'Days_Since_Update', 
                     'Title_Length', 'Title_Word_Count']
        
        # Advanced numerical (từ data loader)
        advanced_nums = ['Log_Price', 'Log_Available', 'Price_Per_Available', 
                        'Price_Times_Available']
        
        # Categorical as numerical
        bucket_features = ['Price_Bucket', 'Available_Bucket']
        
        for col in basic_nums + advanced_nums + bucket_features:
            if col in df.columns:
                numerical_features.append(col)
        
        print(f"   ✅ Tổng {len(numerical_features)} numerical features")
        
        # ========================================
        # 4. BINARY FEATURES
        # ========================================
        binary_features = ['Is_Available', 'Is_High_Price', 'Is_Low_Price',
                          'Is_Recently_Updated', 'Is_Very_Old']
        binary_features = [f for f in binary_features if f in df.columns]
        
        print(f"   ✅ {len(binary_features)} binary features")
        
        # ========================================
        # 5. POLYNOMIAL INTERACTION FEATURES
        # ========================================
        print("\n⚡ Tạo Polynomial Interaction Features...")
        
        # Chỉ tạo interaction cho top important features
        if 'Price' in df.columns and 'Available' in df.columns:
            df['Price_x_Available'] = df['Price'] * df['Available']
            df['Price_div_Available'] = df['Price'] / (df['Available'] + 1)
            df['Price_squared'] = df['Price'] ** 2
            df['Available_squared'] = df['Available'] ** 2
            print(f"   ✅ Tạo 4 interaction features (Price x Available)")
        
        # ========================================
        # 6. GỘP TẤT CẢ FEATURES
        # ========================================
        print("\n📦 Gộp tất cả features...")
        
        feature_cols = []
        
        # Text features
        feature_cols.extend(title_df.columns.tolist())
        
        # Encoded categorical
        for col in categorical_cols:
            if f'{col}_encoded' in df.columns:
                feature_cols.append(f'{col}_encoded')
            if f'{col}_target_encoded' in df.columns:
                feature_cols.append(f'{col}_target_encoded')
        
        # Numerical
        feature_cols.extend(numerical_features)
        
        # Binary
        feature_cols.extend(binary_features)
        
        # Interaction
        interaction_cols = ['Price_x_Available', 'Price_div_Available', 
                          'Price_squared', 'Available_squared']
        for col in interaction_cols:
            if col in df.columns:
                feature_cols.append(col)
        
        # Tạo DataFrame X
        X = pd.concat([
            title_df.reset_index(drop=True),
            df[[col for col in feature_cols if col in df.columns and col not in title_df.columns]].reset_index(drop=True)
        ], axis=1)
        
        # ========================================
        # 7. SCALING (RobustScaler thay vì StandardScaler)
        # ========================================
        print("\n📊 Scaling numerical features...")
        
        scale_cols = numerical_features + interaction_cols
        scale_cols = [col for col in scale_cols if col in X.columns]
        
        if scale_cols:
            X[scale_cols] = self.scaler.fit_transform(X[scale_cols])
            print(f"   ✅ Đã scale {len(scale_cols)} numerical features bằng RobustScaler")
        
        self.feature_names = X.columns.tolist()
        
        # ========================================
        # 8. TARGET VARIABLE
        # ========================================
        if self.use_log_target and 'Sold_Log' in df.columns:
            y = df['Sold_Log']
            print(f"\n🎯 Sử dụng Target: Sold_Log (mean={y.mean():.3f}, std={y.std():.3f})")
        else:
            y = df['Sold']
            print(f"\n🎯 Sử dụng Target: Sold (mean={y.mean():.1f}, std={y.std():.1f})")
        
        self.target_mean = float(y.mean())
        
        # ========================================
        # SUMMARY
        # ========================================
        print("\n" + "="*80)
        print("✅ HOÀN THÀNH FEATURE ENGINEERING")
        print("="*80)
        print(f"📊 Tổng số features: {X.shape[1]}")
        print(f"   • Text (TF-IDF): {len(title_df.columns)}")
        print(f"   • Categorical: {len([c for c in categorical_cols if c in df.columns])}")
        print(f"   • Numerical: {len(numerical_features)}")
        print(f"   • Binary: {len(binary_features)}")
        print(f"   • Interaction: {len([c for c in interaction_cols if c in df.columns])}")
        print(f"📈 Kích thước: X={X.shape}, y={y.shape}")
        print("="*80)
        
        return X, y
    
    def transform_new_input(self, title, brand, perfume_type, price, 
                           available, days_since_update, country='US', state_city='New York'):
        """
        ✅ TRANSFORM INPUT MỚI CẢI TIẾN
        Xử lý unseen labels tốt hơn, tạo đủ features
        """
        
        # 1. Text vectorization
        title_vec = self.title_vectorizer.transform([title])
        title_df = pd.DataFrame(
            title_vec.toarray(),
            columns=[f"title_{name}" for name in self.title_vectorizer.get_feature_names_out()]
        )
        
        # 2. Encode categorical - xử lý unseen labels
        def safe_encode(col_name, value):
            """Xử lý unseen labels bằng 'Unknown'"""
            try:
                return self.label_encoders[col_name].transform([value])[0]
            except (ValueError, KeyError):
                le = self.label_encoders.get(col_name)
                if le is not None:
                    try:
                        return le.transform(['Unknown'])[0]
                    except Exception:
                        return 0
                return 0
        
        def safe_target_encode(col_name, value):
            """Target encoding với fallback"""
            if col_name in self.target_encoders:
                return self.target_encoders[col_name].get(value, self.target_mean or 0)
            return 0
        
        brand_enc = safe_encode('Brand', brand)
        type_enc = safe_encode('Type', perfume_type)
        country_enc = safe_encode('Country', country)
        state_enc = safe_encode('State_City', state_city)
        
        brand_target = safe_target_encode('Brand', brand)
        state_target = safe_target_encode('State_City', state_city)
        
        # 3. Calculate derived features (match training)
        title_length = len(title)
        title_word_count = len(title.split())
        is_available = 1 if available > 0 else 0
        
        # Price features
        log_price = np.log1p(price)
        is_high_price = 1 if price > 50 else 0  # Median approximation
        is_low_price = 1 if price < 25 else 0
        price_bucket = min(4, int(price / 50))  # Simple bucketing
        price_squared = price ** 2
        
        # Available features
        log_available = np.log1p(available)
        available_bucket = min(3, int(available / 25))
        available_squared = available ** 2
        
        # Interaction features
        price_per_available = price / (available + 1)
        price_times_available = price * available
        price_x_available = price * available
        price_div_available = price / (available + 1)
        
        # Time features
        is_recently_updated = 1 if days_since_update < 7 else 0
        is_very_old = 1 if days_since_update > 90 else 0
        
        # 4. Create feature dict
        features = {
            'Brand_encoded': brand_enc,
            'Type_encoded': type_enc,
            'Country_encoded': country_enc,
            'State_City_encoded': state_enc,
            'Price': price,
            'Available': available,
            'Days_Since_Update': days_since_update,
            'Title_Length': title_length,
            'Title_Word_Count': title_word_count,
            'Is_Available': is_available,
            'Log_Price': log_price,
            'Log_Available': log_available,
            'Price_Per_Available': price_per_available,
            'Price_Times_Available': price_times_available,
            'Is_High_Price': is_high_price,
            'Is_Low_Price': is_low_price,
            'Price_Bucket': price_bucket,
            'Available_Bucket': available_bucket,
            'Is_Recently_Updated': is_recently_updated,
            'Is_Very_Old': is_very_old,
            'Price_x_Available': price_x_available,
            'Price_div_Available': price_div_available,
            'Price_squared': price_squared,
            'Available_squared': available_squared,
        }
        
        # Add target encoding if available
        if 'Brand_target_encoded' in self.feature_names:
            features['Brand_target_encoded'] = brand_target
        if 'State_City_target_encoded' in self.feature_names:
            features['State_City_target_encoded'] = state_target
        
        # 5. Scale numerical features
        scale_cols = [
            'Price', 'Available', 'Days_Since_Update', 'Title_Length',
            'Title_Word_Count', 'Log_Price', 'Log_Available',
            'Price_Per_Available', 'Price_Times_Available',
            'Price_Bucket', 'Available_Bucket',
            'Price_x_Available', 'Price_div_Available',
            'Price_squared', 'Available_squared'
        ]
        
        numerical_values = [[features.get(col, 0) for col in scale_cols]]
        scaled_values = self.scaler.transform(numerical_values)[0]
        
        for i, col in enumerate(scale_cols):
            if col in features:
                features[col] = scaled_values[i]
        
        # 6. Combine with title features
        X_new = pd.concat([
            title_df.reset_index(drop=True),
            pd.DataFrame([features])
        ], axis=1)
        
        # 7. Reorder và điền thiếu
        X_new = X_new.reindex(columns=self.feature_names, fill_value=0)
        
        return X_new