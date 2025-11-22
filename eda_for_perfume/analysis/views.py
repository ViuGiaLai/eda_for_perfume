"""
✅ DJANGO VIEWS CẢI TIẾN

Cập nhật để sử dụng improved models
"""

from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views import View
from django.db.models import Avg, Sum, Count
from .models import Perfume, Brand, PredictionHistory

# ✅ Import các class cải tiến
from .ml_models.data_loader import ImprovedPerfumeDataLoader
from .ml_models.feature_engineering import ImprovedFeatureEngineer
from .ml_models.model_training import ImprovedModelTrainer

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
from pathlib import Path


class DashboardView(View):
    """Trang chủ dashboard - Phân tích doanh số"""
    
    def get(self, request):
        perfumes = Perfume.objects.all()
        
        total_sold = perfumes.aggregate(Sum('review_count'))['review_count__sum'] or 0
        avg_sold = perfumes.aggregate(Avg('review_count'))['review_count__avg'] or 0
        
        context = {
            'total_perfumes': perfumes.count(),
            'total_brands': Brand.objects.count(),
            'total_sold': total_sold,
            'avg_sold': round(avg_sold, 1),
            'top_brands': Brand.objects.annotate(
                total_sales=Sum('perfumes__review_count')
            ).order_by('-total_sales')[:5],
            'sales_distribution': self.get_sales_distribution(),
            'price_vs_sales': self.get_price_vs_sales_chart(),
            'brand_sales': self.get_brand_sales_chart(),
            'type_distribution': self.get_type_distribution_pie(),
            'diagnostic_plots': self.get_diagnostic_plots(),
        }
        return render(request, 'analysis/dashboard.html', context)
    
    def get_sales_distribution(self):
        sold_values = list(Perfume.objects.values_list('review_count', flat=True))
        fig = px.histogram(
            x=sold_values, 
            nbins=30,
            title='Phân Bố Doanh Số Bán Hàng',
            labels={'x': 'Số lượng đã bán', 'y': 'Số sản phẩm'}
        )
        return fig.to_html(full_html=False)
    
    def get_price_vs_sales_chart(self):
        data = Perfume.objects.values('price', 'review_count', 'brand__name')
        df = pd.DataFrame(data)
        fig = px.scatter(
            df, 
            x='price', 
            y='review_count',
            color='brand__name',
            title='Mối Quan Hệ Giá - Doanh Số',
            labels={'price': 'Giá (USD)', 'review_count': 'Đã bán'}
        )
        return fig.to_html(full_html=False)
    
    def get_brand_sales_chart(self):
        brands = Brand.objects.annotate(
            total_sales=Sum('perfumes__review_count')
        ).order_by('-total_sales')[:10]
        
        names = [b.name for b in brands]
        sales = [b.total_sales or 0 for b in brands]
        
        fig = go.Figure(data=[go.Bar(x=names, y=sales)])
        fig.update_layout(
            title='Top 10 Thương Hiệu Theo Doanh Số',
            xaxis_title='Thương hiệu',
            yaxis_title='Tổng đã bán'
        )
        return fig.to_html(full_html=False)

    def get_type_distribution_pie(self):
        qs = Perfume.objects.values_list('concentration', flat=True)
        values = [v.strip() if isinstance(v, str) and v.strip() else 'Khác' for v in qs]

        if not values:
            fig = go.Figure(data=[go.Pie(labels=['Không có dữ liệu'], values=[1])])
            fig.update_layout(title='Tỷ lệ các loại nước hoa (nồng độ)')
            return fig.to_html(full_html=False)

        counts = {}
        for v in values:
            counts[v] = counts.get(v, 0) + 1

        labels = list(counts.keys())
        vals = list(counts.values())

        fig = go.Figure(data=[go.Pie(labels=labels, values=vals, hole=0)])
        fig.update_traces(textinfo='percent+label')
        fig.update_layout(title='Tỷ lệ các loại nước hoa (EDT/EDP/Parfum)')
        return fig.to_html(full_html=False)
        
    def get_diagnostic_plots(self):
        """Get paths to diagnostic plots"""
        plots_dir = Path('static/analysis/plots')
        if not plots_dir.exists():
            return []
            
        plot_files = list(plots_dir.glob('*.png'))
        return [f'/static/analysis/plots/{plot.name}' for plot in plot_files]


class PredictView(View):
    """
    ✅ PREDICT VIEW CẢI TIẾN
    Sử dụng improved models và xử lý log transform
    """
    
    @csrf_exempt
    def dispatch(self, request, *args, **kwargs):
        return super().dispatch(request, *args, **kwargs)
    
    def get(self, request):
        brands = Brand.objects.all()
        countries = ['US', 'UK', 'FR', 'DE', 'JP', 'IT', 'CA']
        return render(request, 'analysis/predict.html', {
            'brands': brands, 
            'countries': countries
        })
    
    def post(self, request):
        # Get input parameters
        title = request.POST.get('title')
        brand = request.POST.get('brand')
        perfume_type = request.POST.get('type', 'EDT')
        
        try:
            price = float(request.POST.get('price', 100))
        except Exception:
            price = 100.0
            
        try:
            available = int(float(request.POST.get('available', 50)))
        except Exception:
            available = 50
            
        try:
            days_since_update = int(float(request.POST.get('days_since_update', 0)))
        except Exception:
            days_since_update = 0
            
        country = request.POST.get('country', 'US')
        state_city = request.POST.get('state_city', 'New York')
        
        try:
            # ========================================
            # VALIDATION
            # ========================================
            errors = []
            
            if not title or len(title.strip()) < 3:
                errors.append('Tên sản phẩm phải có ít nhất 3 ký tự.')
            if not brand or not Brand.objects.filter(name=brand).exists():
                errors.append('Thương hiệu không hợp lệ.')
            if perfume_type not in {'EDT', 'EDP', 'Parfum', 'Cologne'}:
                errors.append('Loại/Nồng độ không hợp lệ.')
            if price <= 0 or price > 10000:
                errors.append('Giá phải trong khoảng 0 - 10000 USD.')
            if available < 0 or available > 100000:
                errors.append('Số lượng còn phải trong khoảng 0 - 100000.')
            if days_since_update < 0 or days_since_update > 3650:
                errors.append('Số ngày kể từ lần cập nhật cuối phải trong khoảng 0 - 3650.')
                
            if errors:
                return JsonResponse({'success': False, 'error': ' '.join(errors)})

            # ========================================
            # LOAD MODEL & DATA
            # ========================================
            model = ImprovedModelTrainer.load_model('data/models/perfume_sales_model.pkl')
            
            # ✅ Khởi tạo engineer với use_log_target=True (phải match với training)
            engineer = ImprovedFeatureEngineer(use_log_target=True)
            
            loader = ImprovedPerfumeDataLoader()
            df = loader.load_data()
            df = loader.clean_data(remove_outliers=True, log_transform=True)
            
            # Fit engineer trên toàn bộ data
            X, y = engineer.engineer_features(df)
            
            # ========================================
            # CALCULATE METRICS ON TEST SET
            # ========================================
            from sklearn.model_selection import train_test_split, cross_val_score
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Predict trên test set
            y_test_pred = model.predict(X_test)
            y_train_pred = model.predict(X_train)
            
            # ✅ TÍNH MAPE ĐÚNG (loại bỏ y=0)
            def calculate_mape(y_true, y_pred):
                y_true = np.array(y_true)
                y_pred = np.array(y_pred)
                mask = y_true > 0
                if mask.sum() == 0:
                    return 999.99
                mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
                return min(mape, 999.99)
            
            # Metrics
            test_r2 = r2_score(y_test, y_test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            test_mae = mean_absolute_error(y_test, y_test_pred)
            test_mape = calculate_mape(y_test, y_test_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Overfit gap
            train_r2 = r2_score(y_train, y_train_pred)
            overfit_gap = abs(train_r2 - test_r2)
            
            # ========================================
            # TRANSFORM NEW INPUT
            # ========================================
            X_new = engineer.transform_new_input(
                title, brand, perfume_type, price, 
                available, days_since_update, country, state_city
            )
            
            # ========================================
            # PREDICT
            # ========================================
            raw_pred = float(model.predict(X_new)[0])
            
            # ✅ INVERSE LOG TRANSFORM
            predicted_sales = np.expm1(raw_pred)  # Inverse của log1p
            
            # Clip về khoảng hợp lý
            predicted_sales = int(max(0, np.clip(predicted_sales, 0, 10000)))
            
            # ========================================
            # SAVE PREDICTION HISTORY
            # ========================================
            PredictionHistory.objects.create(
                perfume_name=title,
                input_notes=f"brand={brand} | type={perfume_type} | price={price} | loc={state_city},{country}",
                predicted_rating=float(predicted_sales)
            )

            # ========================================
            # RETURN RESPONSE
            # ========================================
            return JsonResponse({
                'success': True,
                'predicted_sales': predicted_sales,
                'message': f'Dự đoán doanh số cho {title}: {predicted_sales} sản phẩm',
                'metrics': {
                    'mape': float(test_mape),
                    'r2': float(test_r2),
                    'rmse': float(test_rmse),
                    'mae': float(test_mae),
                    'cv_mean': float(cv_mean),
                    'cv_std': float(cv_std),
                    'overfit_gap': float(overfit_gap)
                }
            })
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return JsonResponse({'success': False, 'error': str(e)})


class TrainModelView(View):
    """
    ✅ TRAIN MODEL VIEW CẢI TIẾN
    Sử dụng improved training pipeline
    """
    
    @csrf_exempt
    def dispatch(self, request, *args, **kwargs):
        return super().dispatch(request, *args, **kwargs)
    
    def post(self, request):
        try:
            print("\n" + "="*80)
            print("🚀 BẮT ĐẦU TRAINING MODEL (VIA WEB)")
            print("="*80)
            
            # ========================================
            # 1. LOAD & CLEAN DATA
            # ========================================
            loader = ImprovedPerfumeDataLoader()
            df = loader.load_data()
            df = loader.clean_data(
                remove_outliers=True,   # ✅ Bật xử lý outliers
                log_transform=True      # ✅ Bật log transform
            )
            
            # ========================================
            # 2. FEATURE ENGINEERING
            # ========================================
            engineer = ImprovedFeatureEngineer(use_log_target=True)
            X, y = engineer.engineer_features(df)
            
            # ========================================
            # 3. TRAIN MODELS
            # ========================================
            trainer = ImprovedModelTrainer(use_log_target=True)
            best_model, results = trainer.train_all_models(X, y)
            
            # ========================================
            # 4. SAVE MODEL
            # ========================================
            trainer.save_model('data/models/perfume_sales_model.pkl')
            
            # ========================================
            # 5. GET FEATURE IMPORTANCE
            # ========================================
            feature_importance = trainer.get_feature_importance(X, top_n=15)
            
            print("\n" + "="*80)
            print("✅ HOÀN THÀNH TRAINING")
            print("="*80)
            
            return JsonResponse({
                'success': True,
                'best_model': trainer.best_model_name,
                'results': results.to_dict('records'),
                'feature_importance': (
                    feature_importance.to_dict('records') 
                    if feature_importance is not None else []
                ),
                'config': {
                    'use_log_target': True,
                    'remove_outliers': True,
                    'outliers_removed': loader.outlier_removed_count
                }
            })
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return JsonResponse({'success': False, 'error': str(e)})