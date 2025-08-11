#!/usr/bin/env python3
"""
House Price Prediction using Linear Regression Model
This script trains a linear regression model to predict house prices.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class HousePricePredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='median')
        self.feature_columns = None
        
    def load_data(self, train_path='train.csv', test_path='test.csv'):
        """Load training and test datasets"""
        print("Loading datasets...")
        self.train_data = pd.read_csv(train_path)
        self.test_data = pd.read_csv(test_path)
        
        print(f"Training data shape: {self.train_data.shape}")
        print(f"Test data shape: {self.test_data.shape}")
        
        return self.train_data, self.test_data
    
    def explore_data(self):
        """Basic data exploration"""
        print("\n=== DATA EXPLORATION ===")
        print(f"Training data info:")
        print(f"- Shape: {self.train_data.shape}")
        print(f"- Columns: {len(self.train_data.columns)}")
        print(f"- Missing values: {self.train_data.isnull().sum().sum()}")
        
        # Target variable analysis
        target = 'SalePrice'
        print(f"\nTarget variable ({target}) statistics:")
        print(self.train_data[target].describe())
        
        # Check for outliers in target
        Q1 = self.train_data[target].quantile(0.25)
        Q3 = self.train_data[target].quantile(0.75)
        IQR = Q3 - Q1
        outliers = self.train_data[(self.train_data[target] < Q1 - 1.5*IQR) | 
                                  (self.train_data[target] > Q3 + 1.5*IQR)]
        print(f"- Outliers in target: {len(outliers)} ({len(outliers)/len(self.train_data)*100:.1f}%)")
        
        return self.train_data
    
    def preprocess_data(self, data=None, is_training=True):
        """
        Preprocess the data for training or prediction
        
        Args:
            data: DataFrame to preprocess (if None, uses self.data)
            is_training: Boolean indicating if this is training data
        
        Returns:
            Processed DataFrame and target variable (if training)
        """
        if data is None:
            data = self.train_data.copy() # Changed from self.data to self.train_data
        else:
            data = data.copy()
        
        print("Starting data preprocessing...")
        
        # Handle missing values
        data = self._handle_missing_values(data)
        
        # Feature engineering
        data = self._feature_engineering(data)
        
        # Remove outliers (only for training data)
        if is_training:
            data = self._remove_outliers(data)
        
        # Encode categorical variables
        data = self._encode_categorical(data, is_training)
        
        # Final cleanup - remove any remaining NaN values
        data = data.fillna(0)
        
        # Select features
        if is_training:
            # For training, separate features and target
            if 'SalePrice' in data.columns:
                y = data['SalePrice']
                X = data.drop('SalePrice', axis=1)
            else:
                raise ValueError("SalePrice column not found in training data")
            
            # Store feature columns for later use
            self.feature_columns = X.columns.tolist()
            
            print(f"Preprocessing completed. Features: {len(self.feature_columns)}")
            return X, y
        else:
            # For prediction, ensure all required features are present
            if hasattr(self, 'feature_columns'):
                missing_cols = set(self.feature_columns) - set(data.columns)
                if missing_cols:
                    print(f"Adding missing columns: {missing_cols}")
                    for col in missing_cols:
                        data[col] = 0
                
                # Remove extra columns
                extra_cols = set(data.columns) - set(self.feature_columns)
                if extra_cols:
                    print(f"Removing extra columns: {extra_cols}")
                    data = data.drop(columns=list(extra_cols))
                
                # Reorder columns to match training data
                data = data[self.feature_columns]
            
            print(f"Preprocessing completed for prediction. Features: {len(data.columns)}")
            return data
    
    def _handle_missing_values(self, data):
        """Fill missing values in numeric and categorical columns"""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        categorical_columns = data.select_dtypes(include=['object']).columns
        
        # Fill numeric missing values with median
        data[numeric_columns] = data[numeric_columns].fillna(
            data[numeric_columns].median()
        )
        
        # Fill categorical missing values with mode
        for col in categorical_columns:
            data[col] = data[col].fillna(data[col].mode()[0])
        
        return data
    
    def _feature_engineering(self, data):
        """Create new features from existing ones"""
        # Ensure required columns exist with defaults
        required_cols = {
            'TotalBsmtSF': 0, '1stFlrSF': 0, '2ndFlrSF': 0,
            'FullBath': 0, 'HalfBath': 0, 'BsmtFullBath': 0, 'BsmtHalfBath': 0,
            'YrSold': 2024, 'YearBuilt': 2000, 'YearRemodAdd': 2000,
            'OverallQual': 5, 'OverallCond': 5,
            'GarageYrBlt': 2000,
            'OpenPorchSF': 0, 'EnclosedPorch': 0, '3SsnPorch': 0, 'ScreenPorch': 0,
            'GarageArea': 0, 'Fireplaces': 0, 'PoolArea': 0
        }
        
        for col, default_val in required_cols.items():
            if col not in data.columns:
                data[col] = default_val
        
        # Convert numeric columns to float to ensure proper operations
        numeric_cols = ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'FullBath', 'HalfBath', 
                       'BsmtFullBath', 'BsmtHalfBath', 'YrSold', 'YearBuilt', 'YearRemodAdd',
                       'OverallQual', 'OverallCond', 'GarageYrBlt', 'OpenPorchSF', 
                       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'GarageArea', 
                       'Fireplaces', 'PoolArea']
        
        for col in numeric_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
        
        # Total square footage
        data['TotalSF'] = data['TotalBsmtSF'].astype(float) + data['1stFlrSF'].astype(float) + data['2ndFlrSF'].astype(float)
        
        # Total bathrooms
        data['TotalBathrooms'] = (data['FullBath'].astype(float) + 
                                 0.5 * data['HalfBath'].astype(float) + 
                                 data['BsmtFullBath'].astype(float) + 
                                 0.5 * data['BsmtHalfBath'].astype(float))
        
        # Age of house
        data['HouseAge'] = data['YrSold'].astype(float) - data['YearBuilt'].astype(float)
        data['RemodelAge'] = data['YrSold'].astype(float) - data['YearRemodAdd'].astype(float)
        
        # Quality score (combine overall quality and condition)
        data['QualityScore'] = data['OverallQual'].astype(float) * data['OverallCond'].astype(float)
        
        # Garage age
        data['GarageAge'] = data['YrSold'].astype(float) - data['GarageYrBlt'].astype(float)
        data['GarageAge'] = data['GarageAge'].fillna(0)
        
        # Total porch area
        data['TotalPorchSF'] = (data['OpenPorchSF'].astype(float) + 
                               data['EnclosedPorch'].astype(float) + 
                               data['3SsnPorch'].astype(float) + 
                               data['ScreenPorch'].astype(float))
        
        # Has basement
        data['HasBasement'] = (data['TotalBsmtSF'].astype(float) > 0).astype(int)
        
        # Has garage
        data['HasGarage'] = (data['GarageArea'].astype(float) > 0).astype(int)
        
        # Has fireplace
        data['HasFireplace'] = (data['Fireplaces'].astype(float) > 0).astype(int)
        
        # Has pool
        data['HasPool'] = (data['PoolArea'].astype(float) > 0).astype(int)
        
        return data
    
    def _remove_outliers(self, data):
        """Remove outliers from the target variable"""
        if 'SalePrice' in data.columns:
            y = data['SalePrice']
            Q1 = y.quantile(0.25)
            Q3 = y.quantile(0.75)
            IQR = Q3 - Q1
            outlier_mask = (y >= Q1 - 1.5*IQR) & (y <= Q3 + 1.5*IQR)
            data = data[outlier_mask]
            print(f"Removed {sum(~outlier_mask)} outliers from training data")
        return data

    def _encode_categorical(self, data, is_training=True):
        """Encode categorical variables"""
        categorical_columns = data.select_dtypes(include=['object']).columns
        
        if is_training:
            # Initialize label encoders for training
            self.label_encoders = {}
        
        for col in categorical_columns:
            if is_training:
                # Fit and transform for training data
                self.label_encoders[col] = LabelEncoder()
                data[col] = self.label_encoders[col].fit_transform(data[col])
            else:
                # Transform for prediction data
                if col in self.label_encoders:
                    # Handle unseen values
                    unseen_values = set(data[col].unique()) - set(self.label_encoders[col].classes_)
                    if unseen_values:
                        print(f"Warning: Unseen values in {col}: {unseen_values}")
                        if len(self.label_encoders[col].classes_) > 0:
                            most_common = self.label_encoders[col].classes_[0]
                        else:
                            most_common = data[col].mode()[0]
                        data[col] = data[col].replace(list(unseen_values), most_common)

                    current_values = set(data[col].unique())
                    valid_values = set(self.label_encoders[col].classes_)
                    invalid_values = current_values - valid_values

                    if invalid_values:
                        print(f"Replacing invalid values in {col}: {invalid_values}")
                        data[col] = data[col].replace(list(invalid_values), self.label_encoders[col].classes_[0])

                    data[col] = self.label_encoders[col].transform(data[col])
                else:
                    # If encoder not found, use a simple mapping
                    unique_values = data[col].unique()
                    value_to_int = {val: idx for idx, val in enumerate(unique_values)}
                    data[col] = data[col].map(value_to_int)
        
        return data
    
    def train_model(self):
        """Train the Linear Regression model"""
        print("\n=== TRAINING LINEAR REGRESSION MODEL ===")
        
        # Preprocess data
        X, y = self.preprocess_data(is_training=True)
        
        # Split the data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"R² Score: {r2:.4f}")
        print(f"RMSE: ${rmse:,.2f}")
        print(f"MAE: ${mae:,.2f}")
        
        return r2, rmse, mae
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the model on test set"""
        print("\n=== MODEL EVALUATION ===")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Linear Regression Performance:")
        print(f"  RMSE: ${rmse:,.2f}")
        print(f"  MAE: ${mae:,.2f}")
        print(f"  R²: {r2:.4f}")
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'predictions': y_pred
        }
    
    def get_feature_importance(self):
        """Get feature importance from linear regression coefficients"""
        if not hasattr(self.model, 'coef_'):
            print("Model hasn't been trained yet!")
            return None
        
        # Create feature importance dataframe
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'coefficient': self.model.coef_,
            'importance': np.abs(self.model.coef_)
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def plot_feature_importance(self, top_n=20):
        """Plot feature importance"""
        importance_df = self.get_feature_importance()
        
        if importance_df is None:
            return
        
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(top_n)
        
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance (Absolute Coefficient)')
        plt.title(f'Top {top_n} Features - Linear Regression')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('linear_regression_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict(self, data):
        """Make predictions on new data"""
        # Preprocess the data for prediction
        processed_data = self.preprocess_data(data, is_training=False)
        
        # Make predictions
        predictions = self.model.predict(processed_data)
        
        return predictions
    
    def save_predictions(self, predictions, filename='linear_regression_predictions.csv'):
        """Save predictions to CSV file"""
        submission = pd.DataFrame({
            'Id': range(1, len(predictions) + 1),
            'SalePrice': predictions
        })
        submission.to_csv(filename, index=False)
        print(f"Predictions saved to {filename}")

def main():
    """Main function to run the house price prediction pipeline"""
    print("🏠 HOUSE PRICE PREDICTION USING LINEAR REGRESSION 🏠")
    print("=" * 60)
    
    # Initialize predictor
    predictor = HousePricePredictor()
    
    # Load data
    train_data, test_data = predictor.load_data()
    
    # Explore data
    predictor.explore_data()
    
    # Preprocess data
    train_processed, y = predictor.preprocess_data(is_training=True)
    
    # Split training data
    X_train, X_test, y_train, y_test = train_test_split(
        train_processed, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_test.shape[0]}")
    
    # Train model
    cv_mean, cv_std = predictor.train_model()
    
    # Evaluate model
    evaluation_results = predictor.evaluate_model(X_test, y_test)
    
    print(f"\n🎯 Linear Regression Model Performance:")
    print(f"   Cross-validation R²: {cv_mean:.4f} (+/- {cv_std * 2:.4f})")
    print(f"   Test R²: {evaluation_results['R²']:.4f}")
    
    # Plot feature importance
    predictor.plot_feature_importance(top_n=20)
    
    # Make predictions on test set
    print("\n=== MAKING PREDICTIONS ===")
    test_predictions = predictor.predict(test_data) # Changed from test_processed to test_data
    
    # Save predictions
    predictor.save_predictions(test_predictions, 'linear_regression_predictions.csv')
    
    print(f"\n✅ Linear Regression Prediction completed!")
    print(f"📊 Predicted {len(test_predictions)} house prices")
    print(f"💰 Price range: ${test_predictions.min():,.2f} - ${test_predictions.max():,.2f}")
    print(f"💰 Average predicted price: ${test_predictions.mean():,.2f}")
    
    return predictor, evaluation_results

if __name__ == "__main__":
    predictor, results = main()
