"""
Solar Power Predictive Model
Machine learning model for predicting next-day solar energy generation.

Author: Bryant M.
Date: July 2025
Project: Solar Energy Performance Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import warnings
from pathlib import Path
import json
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

class SolarPowerPredictor:
    """Machine learning model for solar power generation prediction."""
    
    def __init__(self, data_path=None):
        """Initialize the predictor.
        
        Args:
            data_path (str): Path to the processed data file
        """
        self.data_path = data_path or Path("data/processed/combined_solar_data.csv")
        self.output_dir = Path("outputs")
        self.models_dir = self.output_dir / "models"
        self.viz_dir = self.output_dir / "visualizations"
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Model containers
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.best_model = None
        self.feature_importance = None
        self.predictions = {}
        self.model_metrics = {}
        
        # Scalers
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    
    def load_and_prepare_data(self):
        """Load and prepare data for machine learning."""
        print("📥 Loading processed solar data...")
        
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"✅ Loaded {len(self.data):,} records from {self.data_path}")
            
            # Convert datetime
            self.data['DATE_TIME'] = pd.to_datetime(self.data['DATE_TIME'])
            
            # Sort by datetime
            self.data = self.data.sort_values(['PLANT', 'DATE_TIME']).reset_index(drop=True)
            
            print(f"✅ Data prepared: {self.data.shape[0]} rows × {self.data.shape[1]} columns")
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def engineer_features(self):
        """Create features for machine learning model."""
        print("🔧 Engineering features for prediction...")
        
        # Create a copy for feature engineering
        df = self.data.copy()
        
        # Time-based features
        df['YEAR'] = df['DATE_TIME'].dt.year
        df['MONTH'] = df['DATE_TIME'].dt.month
        df['DAY'] = df['DATE_TIME'].dt.day
        df['HOUR'] = df['DATE_TIME'].dt.hour
        df['DAY_OF_YEAR'] = df['DATE_TIME'].dt.dayofyear
        df['WEEK_OF_YEAR'] = df['DATE_TIME'].dt.isocalendar().week
        df['DAY_OF_WEEK'] = df['DATE_TIME'].dt.dayofweek
        df['QUARTER'] = df['DATE_TIME'].dt.quarter
        
        # Cyclical encoding for time features
        df['HOUR_SIN'] = np.sin(2 * np.pi * df['HOUR'] / 24)
        df['HOUR_COS'] = np.cos(2 * np.pi * df['HOUR'] / 24)
        df['DAY_SIN'] = np.sin(2 * np.pi * df['DAY_OF_YEAR'] / 365)
        df['DAY_COS'] = np.cos(2 * np.pi * df['DAY_OF_YEAR'] / 365)
        df['MONTH_SIN'] = np.sin(2 * np.pi * df['MONTH'] / 12)
        df['MONTH_COS'] = np.cos(2 * np.pi * df['MONTH'] / 12)
        
        # Weather-based features
        df['TEMP_DIFF'] = df['MODULE_TEMPERATURE'] - df['AMBIENT_TEMPERATURE']
        df['TEMP_SQUARED'] = df['AMBIENT_TEMPERATURE'] ** 2
        df['IRRADIATION_SQUARED'] = df['IRRADIATION'] ** 2
        
        # Rolling window features (lag features)
        for plant in df['PLANT'].unique():
            plant_mask = df['PLANT'] == plant
            plant_data = df[plant_mask].copy()
            
            # 3-hour rolling averages
            for col in ['AC_POWER', 'AMBIENT_TEMPERATURE', 'IRRADIATION']:
                if col in plant_data.columns:
                    plant_data[f'{col}_3H_AVG'] = plant_data[col].rolling(window=3, min_periods=1).mean()
                    plant_data[f'{col}_3H_STD'] = plant_data[col].rolling(window=3, min_periods=1).std().fillna(0)
            
            # Lag features (previous time periods)
            for col in ['AC_POWER', 'EFFICIENCY', 'IRRADIATION']:
                if col in plant_data.columns:
                    plant_data[f'{col}_LAG_1H'] = plant_data[col].shift(1)
                    plant_data[f'{col}_LAG_3H'] = plant_data[col].shift(3)
                    plant_data[f'{col}_LAG_6H'] = plant_data[col].shift(6)
            
            # Update main dataframe
            df.loc[plant_mask, plant_data.columns] = plant_data
        
        # Encode categorical variables
        df['PLANT_ENCODED'] = self.label_encoder.fit_transform(df['PLANT'])
        
        # Performance ratios and efficiency metrics
        df['DC_AC_RATIO'] = df['AC_POWER'] / (df['DC_POWER'] + 0.001)
        df['POWER_PER_IRRADIATION'] = df['AC_POWER'] / (df['IRRADIATION'] + 0.001)
        df['TEMP_EFFICIENCY_RATIO'] = df['EFFICIENCY'] / (df['MODULE_TEMPERATURE'] + 0.001)
        
        # Clean infinite and NaN values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(df.median(numeric_only=True))
        
        self.engineered_data = df
        print(f"✅ Feature engineering complete: {df.shape[1]} features created")
        return df
    
    def prepare_training_data(self, target_column='AC_POWER', prediction_horizon=1):
        """Prepare data for training predictive models.
        
        Args:
            target_column (str): Column to predict
            prediction_horizon (int): Hours ahead to predict
        """
        print(f"🎯 Preparing training data for {prediction_horizon}h ahead {target_column} prediction...")
        
        df = self.engineered_data.copy()
        
        # Create target variable (next period's power generation)
        df['TARGET'] = df.groupby('PLANT')[target_column].shift(-prediction_horizon)
        
        # Remove rows with missing targets
        df = df.dropna(subset=['TARGET'])
        
        # Define feature columns - exclude non-numeric and identifier columns
        exclude_cols = ['DATE_TIME', 'PLANT', 'SOURCE_KEY', 'TARGET', 'TOTAL_YIELD']
        
        # Get all columns and filter out non-numeric ones
        all_cols = df.columns.tolist()
        feature_cols = []
        
        for col in all_cols:
            if col not in exclude_cols:
                # Check if column is numeric
                try:
                    # Try to convert to numeric, if it fails it's not a good feature
                    pd.to_numeric(df[col], errors='raise')
                    # Also check data type
                    if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                        feature_cols.append(col)
                    elif df[col].dtype == 'object':
                        # Try to convert object columns to numeric
                        try:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                            if not df[col].isnull().all():  # If conversion was successful
                                feature_cols.append(col)
                        except:
                            print(f"   ⚠️  Skipping non-numeric column: {col}")
                except:
                    print(f"   ⚠️  Skipping problematic column: {col}")
        
        print(f"   📊 Selected {len(feature_cols)} numeric features from {len(all_cols)} total columns")
        
        # Split features and target
        X = df[feature_cols].copy()
        y = df['TARGET'].copy()
        
        # Convert all feature columns to numeric (handle any remaining issues)
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Remove any remaining NaN values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        # Final check - remove any infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        print(f"✅ Prepared dataset: {X.shape[0]} samples with {X.shape[1]} features")
        print(f"   Target range: {y.min():.1f} to {y.max():.1f} kW")
        print(f"   Features: {', '.join(X.columns[:5])}{'...' if len(X.columns) > 5 else ''}")
        
        # Store feature names for later use
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and testing sets."""
        print(f"📊 Splitting data (test_size={test_size})...")
        
        # Time-based split to prevent data leakage
        # Use last 20% of data chronologically as test set
        split_idx = int(len(X) * (1 - test_size))
        
        self.X_train = X.iloc[:split_idx]
        self.X_test = X.iloc[split_idx:]
        self.y_train = y.iloc[:split_idx]
        self.y_test = y.iloc[split_idx:]
        
        print(f"✅ Training set: {self.X_train.shape[0]} samples")
        print(f"✅ Test set: {self.X_test.shape[0]} samples")
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_models(self):
        """Train multiple machine learning models."""
        print("🤖 Training machine learning models...")
        
        # Define models to train
        models_config = {
            'Linear_Regression': LinearRegression(),
            'Ridge_Regression': Ridge(alpha=1.0),
            'Random_Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient_Boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        }
        
        # Train each model
        for name, model in models_config.items():
            print(f"   Training {name}...")
            
            try:
                # Use scaled data for linear models, original for tree-based
                if 'Linear' in name or 'Ridge' in name:
                    model.fit(self.X_train_scaled, self.y_train)
                    train_pred = model.predict(self.X_train_scaled)
                    test_pred = model.predict(self.X_test_scaled)
                else:
                    model.fit(self.X_train, self.y_train)
                    train_pred = model.predict(self.X_train)
                    test_pred = model.predict(self.X_test)
                
                # Calculate metrics
                train_r2 = r2_score(self.y_train, train_pred)
                test_r2 = r2_score(self.y_test, test_pred)
                train_mae = mean_absolute_error(self.y_train, train_pred)
                test_mae = mean_absolute_error(self.y_test, test_pred)
                train_rmse = np.sqrt(mean_squared_error(self.y_train, train_pred))
                test_rmse = np.sqrt(mean_squared_error(self.y_test, test_pred))
                
                # Store model and metrics
                self.models[name] = model
                self.predictions[name] = test_pred
                self.model_metrics[name] = {
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'accuracy_percentage': test_r2 * 100
                }
                
                print(f"      ✅ {name}: R² = {test_r2:.3f}, MAE = {test_mae:.1f} kW")
                
            except Exception as e:
                print(f"      ❌ Failed to train {name}: {e}")
        
        # Select best model based on test R²
        best_model_name = max(self.model_metrics.keys(), 
                            key=lambda x: self.model_metrics[x]['test_r2'])
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name
        
        print(f"\n🏆 Best model: {best_model_name} (R² = {self.model_metrics[best_model_name]['test_r2']:.3f})")
        
        return self.models
    
    def evaluate_models(self):
        """Comprehensive model evaluation and comparison."""
        print("\n📈 Evaluating model performance...")
        
        # Create comparison visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Solar Power Prediction Model Evaluation', fontsize=16, fontweight='bold')
        
        # 1. Model comparison metrics
        model_names = list(self.model_metrics.keys())
        r2_scores = [self.model_metrics[name]['test_r2'] for name in model_names]
        mae_scores = [self.model_metrics[name]['test_mae'] for name in model_names]
        
        axes[0, 0].bar(model_names, r2_scores, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Model Accuracy Comparison (R² Score)')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add accuracy percentage labels
        for i, (name, r2) in enumerate(zip(model_names, r2_scores)):
            axes[0, 0].text(i, r2 + 0.02, f'{r2*100:.1f}%', ha='center', fontweight='bold')
        
        # 2. Mean Absolute Error comparison
        axes[0, 1].bar(model_names, mae_scores, alpha=0.7, color='lightcoral')
        axes[0, 1].set_title('Model Error Comparison (MAE)')
        axes[0, 1].set_ylabel('Mean Absolute Error (kW)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Actual vs Predicted for best model
        best_pred = self.predictions[self.best_model_name]
        axes[1, 0].scatter(self.y_test, best_pred, alpha=0.6, s=30)
        axes[1, 0].plot([self.y_test.min(), self.y_test.max()], 
                       [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[1, 0].set_title(f'Actual vs Predicted ({self.best_model_name})')
        axes[1, 0].set_xlabel('Actual Power (kW)')
        axes[1, 0].set_ylabel('Predicted Power (kW)')
        
        # Add R² to plot
        r2_text = f'R² = {self.model_metrics[self.best_model_name]["test_r2"]:.3f}'
        axes[1, 0].text(0.05, 0.95, r2_text, transform=axes[1, 0].transAxes, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 4. Residual plot for best model
        residuals = self.y_test - best_pred
        axes[1, 1].scatter(best_pred, residuals, alpha=0.6, s=30)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_title(f'Residual Plot ({self.best_model_name})')
        axes[1, 1].set_xlabel('Predicted Power (kW)')
        axes[1, 1].set_ylabel('Residuals (kW)')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / "04_model_evaluation.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Feature importance for tree-based models
        if hasattr(self.best_model, 'feature_importances_'):
            self.plot_feature_importance()
        
        # Save model metrics
        with open(self.models_dir / "model_metrics.json", 'w') as f:
            json.dump(self.model_metrics, f, indent=2)
        
        print("✅ Model evaluation complete")
        return self.model_metrics
    
    def plot_feature_importance(self):
        """Plot feature importance for the best model."""
        if not hasattr(self.best_model, 'feature_importances_'):
            return
        
        # Get feature importance
        importance = self.best_model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Plot top 20 features
        plt.figure(figsize=(12, 8))
        top_features = feature_importance_df.head(20)
        
        plt.barh(range(len(top_features)), top_features['importance'], alpha=0.7)
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top 20 Feature Importance ({self.best_model_name})')
        plt.gca().invert_yaxis()
        
        # Add importance values as text
        for i, importance in enumerate(top_features['importance']):
            plt.text(importance + 0.001, i, f'{importance:.3f}', va='center')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / "05_feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save feature importance
        feature_importance_df.to_csv(self.models_dir / "feature_importance.csv", index=False)
        self.feature_importance = feature_importance_df
        
        print("✅ Feature importance analysis complete")
    
    def hyperparameter_tuning(self, model_name='Random_Forest'):
        """Perform hyperparameter tuning on the best performing model."""
        print(f"🔧 Hyperparameter tuning for {model_name}...")
        
        if model_name == 'Random_Forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
            
        elif model_name == 'Gradient_Boosting':
            param_grid = {
                'n_estimators': [50, 100, 150],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.8, 0.9, 1.0]
            }
            base_model = GradientBoostingRegressor(random_state=42)
        else:
            print(f"Hyperparameter tuning not implemented for {model_name}")
            return None
        
        # Perform grid search
        grid_search = GridSearchCV(
            base_model, 
            param_grid, 
            cv=5, 
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        # Update best model
        tuned_model = grid_search.best_estimator_
        tuned_pred = tuned_model.predict(self.X_test)
        tuned_r2 = r2_score(self.y_test, tuned_pred)
        
        print(f"✅ Tuned model R²: {tuned_r2:.3f}")
        print(f"✅ Best parameters: {grid_search.best_params_}")
        
        # Update if better
        if tuned_r2 > self.model_metrics[self.best_model_name]['test_r2']:
            self.best_model = tuned_model
            self.best_model_name = f"{model_name}_Tuned"
            print(f"🏆 New best model: {self.best_model_name}")
        
        return tuned_model
    
    def save_model(self):
        """Save the best trained model."""
        print("💾 Saving best model...")
        
        model_filename = self.models_dir / f"solar_power_predictor_{self.best_model_name}.pkl"
        joblib.dump(self.best_model, model_filename)
        
        # Save scaler
        scaler_filename = self.models_dir / "feature_scaler.pkl"
        joblib.dump(self.scaler, scaler_filename)
        
        # Save feature names and model info
        model_info = {
            'model_name': self.best_model_name,
            'feature_names': self.feature_names,
            'accuracy': self.model_metrics[self.best_model_name]['test_r2'],
            'mae': self.model_metrics[self.best_model_name]['test_mae'],
            'training_date': datetime.now().isoformat(),
            'model_file': str(model_filename),
            'scaler_file': str(scaler_filename)
        }
        
        with open(self.models_dir / "model_info.json", 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"✅ Model saved: {model_filename}")
        print(f"✅ Model info saved: {self.models_dir / 'model_info.json'}")
        
        return model_filename
    
    def predict_next_day(self, input_features):
        """Make predictions for next day solar power generation."""
        if self.best_model is None:
            print("❌ No trained model available")
            return None
        
        # Ensure input has correct features
        if isinstance(input_features, dict):
            input_df = pd.DataFrame([input_features])
        else:
            input_df = input_features
        
        # Make prediction
        if 'Linear' in self.best_model_name or 'Ridge' in self.best_model_name:
            prediction = self.best_model.predict(self.scaler.transform(input_df[self.feature_names]))
        else:
            prediction = self.best_model.predict(input_df[self.feature_names])
        
        return prediction[0] if len(prediction) == 1 else prediction
    
    def run_complete_modeling(self):
        """Execute the complete predictive modeling pipeline."""
        print("🤖 STARTING SOLAR POWER PREDICTIVE MODELING")
        print("=" * 60)
        
        # Load and prepare data
        if not self.load_and_prepare_data():
            return False
        
        # Engineer features
        self.engineer_features()
        
        # Prepare training data
        X, y = self.prepare_training_data()
        
        # Split data
        self.split_data(X, y)
        
        # Train models
        self.train_models()
        
        # Evaluate models
        self.evaluate_models()
        
        # Hyperparameter tuning for best model
        if self.best_model_name in ['Random_Forest', 'Gradient_Boosting']:
            self.hyperparameter_tuning(self.best_model_name)
        
        # Save model
        self.save_model()
        
        # Print final results
        best_metrics = self.model_metrics[self.best_model_name]
        print(f"\n🎉 PREDICTIVE MODELING COMPLETE!")
        print(f"✅ Best model: {self.best_model_name}")
        print(f"✅ Accuracy: {best_metrics['accuracy_percentage']:.1f}%")
        print(f"✅ Mean Absolute Error: {best_metrics['test_mae']:.1f} kW")
        print(f"✅ Model saved for future predictions")
        
        return True

def main():
    """Main execution function."""
    data_path = Path("data/processed/combined_solar_data.csv")
    
    if not data_path.exists():
        print(f"❌ Processed data not found: {data_path}")
        print("Please run the main analysis script first.")
        return 1
    
    predictor = SolarPowerPredictor(data_path)
    
    if predictor.run_complete_modeling():
        print("\n🚀 Next steps:")
        print("1. Review model evaluation plots in outputs/visualizations/")
        print("2. Check model metrics in outputs/models/")
        print("3. Use saved model for real-time predictions")
        return 0
    else:
        print("❌ Predictive modeling failed.")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())