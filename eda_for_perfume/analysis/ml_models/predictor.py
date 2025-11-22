"""
✅ PERFUME SALES PREDICTOR CẢI TIẾN

Thay thế các file cũ:
- data_loader.py → improved_data_loader.py
- feature_engineering.py → improved_feature_engineering.py  
- model_training.py → improved_model_training.py

Cải tiến chính:
1. Xử lý outliers bằng IQR
2. Log transform cho target variable
3. RobustScaler thay StandardScaler
4. Target encoding cho categorical
5. Polynomial interaction features
6. Hyperparameter tuning tốt hơn
7. MAPE tính đúng (loại bỏ y=0)
"""

# Import các class cải tiến
from improved_data_loader import ImprovedPerfumeDataLoader
from improved_feature_engineering import ImprovedFeatureEngineer
from improved_model_training import ImprovedModelTrainer
import numpy as np

class ImprovedPerfumeSalesPredictor:
    """
    ✅ MAIN CLASS CẢI TIẾN
    Tích hợp tất cả improvements
    """
    
    def __init__(self, use_log_target=True, remove_outliers=True):
        """
        Parameters:
        -----------
        use_log_target : bool
            Sử dụng log transform cho target (khuyến nghị: True)
        remove_outliers : bool
            Loại bỏ outliers bằng IQR (khuyến nghị: True)
        """
        self.data_loader = ImprovedPerfumeDataLoader()
        self.feature_engineer = ImprovedFeatureEngineer(use_log_target=use_log_target)
        self.model_trainer = ImprovedModelTrainer(use_log_target=use_log_target)
        
        self.use_log_target = use_log_target
        self.remove_outliers = remove_outliers
        self.model = None
        self.is_trained = False
    
    def train(self, csv_path=None):
        """
        ✅ TRAIN MODEL VỚI PIPELINE CẢI TIẾN
        """
        print("\n" + "="*80)
        print("🎯 BẮT ĐẦU TRAINING MODEL DỰ ĐOÁN DOANH SỐ (IMPROVED VERSION)")
        print("="*80)
        print(f"\n⚙️  Cấu hình:")
        print(f"   • Use Log Target: {self.use_log_target}")
        print(f"   • Remove Outliers: {self.remove_outliers}")
        
        # ========================================
        # 1. LOAD & CLEAN DATA
        # ========================================
        if csv_path:
            self.data_loader.csv_path = csv_path
        
        df = self.data_loader.load_data()
        df = self.data_loader.clean_data(
            remove_outliers=self.remove_outliers,
            log_transform=self.use_log_target
        )
        
        # Show statistics
        self.data_loader.get_statistics()
        
        # ========================================
        # 2. FEATURE ENGINEERING
        # ========================================
        X, y = self.feature_engineer.engineer_features(df)
        
        # ========================================
        # 3. TRAIN MODELS
        # ========================================
        self.model, results = self.model_trainer.train_all_models(X, y)
        
        # ========================================
        # 4. FEATURE IMPORTANCE
        # ========================================
        feature_importance = self.model_trainer.get_feature_importance(X, top_n=20)
        
        # ========================================
        # 5. SAVE MODEL
        # ========================================
        self.model_trainer.save_model()
        
        self.is_trained = True
        
        print("\n" + "="*80)
        print("✅ HOÀN TẤT TRAINING MODEL")
        print("="*80)
        
        return {
            'results': results,
            'feature_importance': feature_importance,
            'best_model': self.model_trainer.best_model_name,
            'config': {
                'use_log_target': self.use_log_target,
                'remove_outliers': self.remove_outliers,
                'outliers_removed': self.data_loader.outlier_removed_count
            }
        }
    
    def predict(self, title, brand, perfume_type='EDT', price=100, 
                available=50, days_since_update=0, country='US', state_city='New York'):
        """
        ✅ DỰ ĐOÁN VỚI INVERSE TRANSFORM (nếu dùng log)
        """
        # Load model nếu chưa train
        if not self.is_trained:
            try:
                self.model = self.model_trainer.load_model()
                self.is_trained = True
                print("✅ Đã load model từ file")
            except FileNotFoundError:
                raise Exception("❌ Model chưa được train. Vui lòng train model trước!")
        
        # Transform input
        X_new = self.feature_engineer.transform_new_input(
            title, brand, perfume_type, price, 
            available, days_since_update, country, state_city
        )
        
        # Predict
        prediction = self.model.predict(X_new)[0]
        
        # ✅ INVERSE TRANSFORM nếu dùng log
        if self.use_log_target:
            prediction = np.expm1(prediction)  # Inverse của log1p
        
        # Clip về khoảng hợp lý (>= 0)
        prediction = max(0, round(prediction))
        
        return prediction
    
    def batch_predict(self, data_list):
        """Predict cho nhiều sản phẩm"""
        predictions = []
        for data in data_list:
            pred = self.predict(**data)
            predictions.append(pred)
        return predictions
    
    def get_model_metrics(self):
        """
        ✅ LẤY METRICS CỦA BEST MODEL
        Dùng để hiển thị trên web UI
        """
        if not self.model_trainer.results:
            return None
            
        import pandas as pd
        results_df = pd.DataFrame(self.model_trainer.results)
        best_result = results_df[results_df['Model'] == self.model_trainer.best_model_name].iloc[0]
        
        return {
            'model_name': self.model_trainer.best_model_name,
            'test_r2': float(best_result['Test_R2']),
            'test_rmse': float(best_result['Test_RMSE']),
            'test_mae': float(best_result['Test_MAE']),
            'test_mape': float(best_result['Test_MAPE']),
            'cv_mean': float(best_result['CV_Mean']),
            'cv_std': float(best_result['CV_Std']),
            'overfit_gap': float(best_result['Overfit_Gap'])
        }


# ========================================
# EXAMPLE USAGE
# ========================================
if __name__ == "__main__":
    print("="*80)
    print("🚀 PERFUME SALES PREDICTOR - IMPROVED VERSION")
    print("="*80)
    
    # Initialize predictor với cấu hình tối ưu
    predictor = ImprovedPerfumeSalesPredictor(
        use_log_target=True,      # ✅ Bật log transform
        remove_outliers=True      # ✅ Bật xử lý outliers
    )
    
    # Train model
    results = predictor.train('analysis/ebay_mens_perfume.csv')
    
    # Print summary
    print("\n" + "="*80)
    print("📋 TRAINING SUMMARY")
    print("="*80)
    print(f"✅ Best Model: {results['best_model']}")
    print(f"✅ Configuration:")
    print(f"   • Log Transform: {results['config']['use_log_target']}")
    print(f"   • Outliers Removed: {results['config']['outliers_removed']}")
    
    # Get metrics
    metrics = predictor.get_model_metrics()
    if metrics:
        print(f"\n📊 Best Model Metrics:")
        print(f"   • Test R²: {metrics['test_r2']:.4f}")
        print(f"   • Test RMSE: {metrics['test_rmse']:.2f}")
        print(f"   • Test MAE: {metrics['test_mae']:.2f}")
        print(f"   • Test MAPE: {metrics['test_mape']:.2f}%")
        print(f"   • CV Score: {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
        print(f"   • Overfit Gap: {metrics['overfit_gap']:.4f}")
    
    # ========================================
    # TEST PREDICTIONS
    # ========================================
    print("\n" + "="*80)
    print("🔮 DỰ ĐOÁN DOANH SỐ - TEST CASES")
    print("="*80)
    
    test_cases = [
        {
            'title': 'Dior Sauvage EDT 100ml Fresh Spicy',
            'brand': 'Dior',
            'perfume_type': 'EDT',
            'price': 89.99,
            'available': 100,
            'days_since_update': 2,
            'country': 'US',
            'state_city': 'New York'
        },
        {
            'title': 'Chanel Bleu de Chanel EDP Woody Aromatic',
            'brand': 'Chanel',
            'perfume_type': 'EDP',
            'price': 120.00,
            'available': 50,
            'days_since_update': 5,
            'country': 'UK',
            'state_city': 'London'
        },
        {
            'title': 'Budget EDT Fresh Citrus',
            'brand': 'Generic',
            'perfume_type': 'EDT',
            'price': 25.00,
            'available': 200,
            'days_since_update': 30,
            'country': 'US',
            'state_city': 'Los Angeles'
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        pred = predictor.predict(**case)
        print(f"\n{i}. {case['title']}")
        print(f"   📦 Brand: {case['brand']} | Type: {case['perfume_type']}")
        print(f"   💰 Price: ${case['price']:.2f} | Available: {case['available']}")
        print(f"   📍 Location: {case['state_city']}, {case['country']}")
        print(f"   ⏰ Days since update: {case['days_since_update']}")
        print(f"   📊 Dự đoán doanh số: {pred} sản phẩm")
    
    print("\n" + "="*80)
    print("✅ HOÀN THÀNH")
    print("="*80)