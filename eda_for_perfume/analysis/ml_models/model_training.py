from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import numpy as np
import pandas as pd
import os

class ImprovedModelTrainer:
    """
    ✅ MODEL TRAINING CẢI TIẾN với:
    - Thêm ElasticNet, ExtraTreesRegressor
    - Hyperparameter tuning tốt hơn
    - Validation strategy chặt chẽ hơn
    - MAPE tính đúng (loại bỏ y=0)
    """
    
    def __init__(self, use_log_target=True):
        self.use_log_target = use_log_target
        self.models = {
            'LinearRegression': LinearRegression(),
            
            'Ridge': Ridge(
                alpha=50.0,  # Tăng từ 10 → 50 để regularization mạnh hơn
                random_state=42
            ),
            
            'Lasso': Lasso(
                alpha=5.0,  # Tăng từ 1 → 5
                random_state=42,
                max_iter=5000
            ),
            
            'ElasticNet': ElasticNet(  # ✅ NEW MODEL
                alpha=5.0,
                l1_ratio=0.5,  # Mix L1 + L2
                random_state=42,
                max_iter=5000
            ),
            
            'DecisionTree': DecisionTreeRegressor(
                max_depth=8,  # Giảm từ 10 → 8 để tránh overfit
                min_samples_split=10,  # Tăng từ 2 → 10
                min_samples_leaf=5,  # Thêm constraint
                random_state=42
            ),
            
            'RandomForest': RandomForestRegressor(
                n_estimators=200,  # Số cây đủ lớn
                max_depth=8,  # Giảm sâu cây để giảm overfit
                min_samples_split=20,  # Yêu cầu nhiều mẫu hơn trước khi split
                min_samples_leaf=8,  # Lá phải có nhiều mẫu hơn
                max_features='sqrt',  # Hạn chế số feature mỗi split
                max_samples=0.8,  # Chỉ dùng 80% mẫu cho mỗi cây
                random_state=42,
                n_jobs=-1
            ),
            
            'ExtraTrees': ExtraTreesRegressor(  # ✅ NEW MODEL
                n_estimators=200,
                max_depth=8,
                min_samples_split=20,
                min_samples_leaf=8,
                max_features='sqrt',
                max_samples=0.8,
                random_state=42,
                n_jobs=-1
            ),
            
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=200,  # Nhiều cây hơn nhưng mỗi bước học chậm hơn
                max_depth=3,  # Cây nông hơn để bớt overfit
                learning_rate=0.03,  # Học chậm hơn để tổng thể mượt hơn
                min_samples_split=20,
                min_samples_leaf=8,
                subsample=0.7,  # Dùng 70% mẫu mỗi cây để tăng regularization
                random_state=42
            )
        }
        
        self.best_model = None
        self.best_model_name = None
        self.results = []
        
    def calculate_mape(self, y_true, y_pred):
        """
        ✅ TÍNH MAPE ĐÚNG
        Loại bỏ các giá trị y_true = 0 để tránh division by zero
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Chỉ tính MAPE trên các giá trị > 0
        mask = y_true > 0
        
        if mask.sum() == 0:
            return 999.99  # Không có giá trị nào > 0
        
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        # Cap ở 999% để tránh số quá lớn
        return min(mape, 999.99)
    
    def train_all_models(self, X, y, test_size=0.2, cv_folds=5):
        """
        ✅ TRAIN TẤT CẢ MODELS VỚI VALIDATION STRATEGY TỐT HƠN
        """
        print("\n" + "="*80)
        print("🚀 BẮT ĐẦU TRAINING MODELS (IMPROVED VERSION)")
        print("="*80)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=True
        )
        
        print(f"\n📊 Chia dữ liệu:")
        print(f"   • Train set: {X_train.shape[0]} samples")
        print(f"   • Test set: {X_test.shape[0]} samples")
        print(f"   • Features: {X_train.shape[1]}")
        
        # Setup cross-validation
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        best_score = -np.inf
        
        for name, model in self.models.items():
            print(f"\n{'='*60}")
            print(f"📝 Training: {name}")
            print(f"{'='*60}")
            
            try:
                # Train
                model.fit(X_train, y_train)
                
                # Predict
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
                # ✅ METRICS ĐÚNG
                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)
                
                train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                
                train_mae = mean_absolute_error(y_train, y_train_pred)
                test_mae = mean_absolute_error(y_test, y_test_pred)
                
                # ✅ MAPE đúng (loại bỏ y=0)
                test_mape = self.calculate_mape(y_test, y_test_pred)
                train_mape = self.calculate_mape(y_train, y_train_pred)
                
                # Cross-validation với KFold
                cv_scores = cross_val_score(
                    model, X_train, y_train, 
                    cv=kfold, 
                    scoring='r2', 
                    n_jobs=-1
                )
                
                # ✅ OVERFIT GAP đúng
                overfit_gap = abs(train_r2 - test_r2)
                
                # Save results
                result = {
                    'Model': name,
                    'Train_R2': train_r2,
                    'Test_R2': test_r2,
                    'Train_RMSE': train_rmse,
                    'Test_RMSE': test_rmse,
                    'Train_MAE': train_mae,
                    'Test_MAE': test_mae,
                    'Train_MAPE': train_mape,
                    'Test_MAPE': test_mape,
                    'CV_Mean': cv_scores.mean(),
                    'CV_Std': cv_scores.std(),
                    'CV_Min': cv_scores.min(),
                    'CV_Max': cv_scores.max(),
                    'Overfit_Gap': overfit_gap
                }
                
                self.results.append(result)
                
                # Print metrics
                print(f"\n📊 Performance Metrics:")
                print(f"   • Train R²: {train_r2:.4f}")
                print(f"   • Test R²:  {test_r2:.4f}")
                print(f"   • Overfit Gap: {overfit_gap:.4f}", 
                      "✅" if overfit_gap < 0.1 else "⚠️" if overfit_gap < 0.2 else "❌")
                
                print(f"\n   • Train RMSE: {train_rmse:.2f}")
                print(f"   • Test RMSE:  {test_rmse:.2f}")
                
                print(f"\n   • Train MAE: {train_mae:.2f}")
                print(f"   • Test MAE:  {test_mae:.2f}")
                
                print(f"\n   • Train MAPE: {train_mape:.2f}%", 
                      "✅" if train_mape < 20 else "⚠️" if train_mape < 50 else "❌")
                print(f"   • Test MAPE:  {test_mape:.2f}%", 
                      "✅" if test_mape < 20 else "⚠️" if test_mape < 50 else "❌")
                
                print(f"\n   • CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                print(f"   • CV Range: [{cv_scores.min():.4f}, {cv_scores.max():.4f}]")
                
                # ✅ CHỌN MODEL TỐT NHẤT DựA trên Test R² và Overfit Gap
                # Ưu tiên model có Test R² cao và Overfit Gap thấp
                # score = test_r2 - (overfit_gap * 0.5)  # Penalty cho overfit
                
                # ✅CHỌN MODEL TỐT NHẤT DựA trên Test MAPE + Test R² + Overfit Gap
                # - Ưu tiên MAPE thấp (quan trọng nhất)
                # - Vẫn khuyến khích Test R² cao
                # - Phạt mô hình overfit (Overfit_Gap lớn)
                score = -(test_mape / 100.0) + 0.2 * test_r2 - 0.5 * overfit_gap

                if score > best_score:
                    best_score = score
                    self.best_model = model
                    self.best_model_name = name
                    print(f"\n   🏆 NEW BEST MODEL! (Score: {score:.4f})")
                    
            except Exception as e:
                print(f"\n   ❌ Error training {name}: {str(e)}")
                continue
        
        # ========================================
        # COMPARISON TABLE
        # ========================================
        print("\n" + "="*80)
        print("📊 BẢNG SO SÁNH TẤT CẢ MODELS")
        print("="*80)
        
        results_df = pd.DataFrame(self.results)
        
        # Sort by Test R² descending
        results_df = results_df.sort_values('Test_R2', ascending=False)
        
        # Format for display
        display_df = results_df.copy()
        for col in ['Train_R2', 'Test_R2', 'CV_Mean']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
        for col in ['Train_RMSE', 'Test_RMSE', 'Train_MAE', 'Test_MAE']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}")
        for col in ['Train_MAPE', 'Test_MAPE', 'CV_Std', 'Overfit_Gap']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}")
        
        print(display_df.to_string(index=False))
        
        # ========================================
        # BEST MODEL SUMMARY
        # ========================================
        print("\n" + "="*80)
        print("🏆 MODEL TỐT NHẤT")
        print("="*80)
        
        best_result = results_df[results_df['Model'] == self.best_model_name].iloc[0]
        
        print(f"\n🎯 Model: {self.best_model_name}")
        print(f"\n📊 Test Set Performance:")
        print(f"   • R² Score: {best_result['Test_R2']:.4f}")
        print(f"   • RMSE: {best_result['Test_RMSE']:.2f}")
        print(f"   • MAE: {best_result['Test_MAE']:.2f}")
        print(f"   • MAPE: {best_result['Test_MAPE']:.2f}%")
        
        print(f"\n🔄 Cross-Validation:")
        print(f"   • Mean R²: {best_result['CV_Mean']:.4f}")
        print(f"   • Std: ±{best_result['CV_Std']:.4f}")
        
        print(f"\n⚖️  Overfit Assessment:")
        print(f"   • Gap: {best_result['Overfit_Gap']:.4f}")
        if best_result['Overfit_Gap'] < 0.1:
            print(f"   • Status: ✅ Cân bằng tốt")
        elif best_result['Overfit_Gap'] < 0.2:
            print(f"   • Status: ⚠️  Hơi overfit")
        else:
            print(f"   • Status: ❌ Overfit nghiêm trọng")
        
        print("="*80)
        
        return self.best_model, results_df
    
    def get_feature_importance(self, X, top_n=20):
        """Lấy top features quan trọng nhất"""
        if self.best_model_name in ['RandomForest', 'GradientBoosting', 
                                     'DecisionTree', 'ExtraTrees']:
            importances = self.best_model.feature_importances_
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': importances
            }).sort_values('Importance', ascending=False).head(top_n)
            
            print(f"\n" + "="*80)
            print(f"🔝 TOP {top_n} FEATURES QUAN TRỌNG NHẤT ({self.best_model_name})")
            print("="*80)
            
            for idx, row in feature_importance.iterrows():
                bar_length = int(row['Importance'] * 50)
                bar = '█' * bar_length
                print(f"   {row['Feature'][:40]:<40} {bar} {row['Importance']:.4f}")
            
            print("="*80)
            
            return feature_importance
        else:
            print(f"\n⚠️  Model {self.best_model_name} không hỗ trợ feature importance")
            return None
    
    def save_model(self, filepath='data/models/perfume_sales_model.pkl'):
        """Lưu model đã train"""
        if not os.path.isabs(filepath):
            filepath = os.path.join(os.path.dirname(__file__), '..', '..', filepath)
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.best_model, filepath)
        print(f"\n💾 Đã lưu model tại: {filepath}")
    
    @staticmethod
    def load_model(filepath='data/models/perfume_sales_model.pkl'):
        """Load model đã train"""
        if not os.path.isabs(filepath):
            filepath = os.path.join(os.path.dirname(__file__), '..', '..', filepath)
        
        return joblib.load(filepath)