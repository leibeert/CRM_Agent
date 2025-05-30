"""
Machine Learning Models for Candidate-Job Matching

This module contains the ML models that predict candidate-job compatibility
based on extracted features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)


class CandidateJobMatcher:
    """
    ML model for predicting candidate-job compatibility scores.
    """
    
    def __init__(self, model_type: str = 'ensemble'):
        """
        Initialize the matcher with specified model type.
        
        Args:
            model_type: Type of model ('random_forest', 'gradient_boost', 'neural_network', 'ensemble')
        """
        self.model_type = model_type
        self.models = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_importance = None
        self.training_metrics = {}
        
        # Initialize models based on type
        if model_type == 'random_forest':
            self.models['main'] = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boost':
            self.models['main'] = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        elif model_type == 'neural_network':
            self.models['main'] = MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            )
        elif model_type == 'ensemble':
            # Ensemble of multiple models
            self.models = {
                'random_forest': RandomForestRegressor(
                    n_estimators=100,
                    max_depth=15,
                    min_samples_split=5,
                    random_state=42,
                    n_jobs=-1
                ),
                'gradient_boost': GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                ),
                'neural_network': MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    activation='relu',
                    solver='adam',
                    alpha=0.001,
                    max_iter=300,
                    random_state=42
                ),
                'linear': Ridge(alpha=1.0)
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              validation_split: float = 0.2,
              optimize_hyperparameters: bool = False) -> Dict[str, float]:
        """
        Train the model(s) on the provided data.
        
        Args:
            X: Feature matrix
            y: Target scores (compatibility scores)
            validation_split: Fraction of data to use for validation
            optimize_hyperparameters: Whether to perform hyperparameter optimization
            
        Returns:
            Dictionary of training metrics
        """
        logger.info(f"Training {self.model_type} model with {X.shape[0]} samples and {X.shape[1]} features")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        metrics = {}
        
        if self.model_type == 'ensemble':
            # Train ensemble models
            ensemble_predictions = []
            
            for name, model in self.models.items():
                logger.info(f"Training {name} model...")
                
                if optimize_hyperparameters and name in ['random_forest', 'gradient_boost']:
                    model = self._optimize_hyperparameters(model, X_train_scaled, y_train)
                    self.models[name] = model
                
                # Train model
                if name == 'neural_network':
                    # Neural network needs scaled data
                    model.fit(X_train_scaled, y_train)
                    val_pred = model.predict(X_val_scaled)
                else:
                    # Tree-based models can use original data
                    model.fit(X_train, y_train)
                    val_pred = model.predict(X_val)
                
                ensemble_predictions.append(val_pred)
                
                # Calculate metrics for individual model
                mse = mean_squared_error(y_val, val_pred)
                r2 = r2_score(y_val, val_pred)
                mae = mean_absolute_error(y_val, val_pred)
                
                metrics[f'{name}_mse'] = mse
                metrics[f'{name}_r2'] = r2
                metrics[f'{name}_mae'] = mae
                
                logger.info(f"{name} - MSE: {mse:.4f}, R2: {r2:.4f}, MAE: {mae:.4f}")
            
            # Ensemble prediction (simple average)
            ensemble_pred = np.mean(ensemble_predictions, axis=0)
            
            # Ensemble metrics
            ensemble_mse = mean_squared_error(y_val, ensemble_pred)
            ensemble_r2 = r2_score(y_val, ensemble_pred)
            ensemble_mae = mean_absolute_error(y_val, ensemble_pred)
            
            metrics['ensemble_mse'] = ensemble_mse
            metrics['ensemble_r2'] = ensemble_r2
            metrics['ensemble_mae'] = ensemble_mae
            
            logger.info(f"Ensemble - MSE: {ensemble_mse:.4f}, R2: {ensemble_r2:.4f}, MAE: {ensemble_mae:.4f}")
            
        else:
            # Train single model
            model = self.models['main']
            
            if optimize_hyperparameters:
                model = self._optimize_hyperparameters(model, X_train_scaled, y_train)
                self.models['main'] = model
            
            if self.model_type == 'neural_network':
                model.fit(X_train_scaled, y_train)
                val_pred = model.predict(X_val_scaled)
            else:
                model.fit(X_train, y_train)
                val_pred = model.predict(X_val)
            
            # Calculate metrics
            mse = mean_squared_error(y_val, val_pred)
            r2 = r2_score(y_val, val_pred)
            mae = mean_absolute_error(y_val, val_pred)
            
            metrics['mse'] = mse
            metrics['r2'] = r2
            metrics['mae'] = mae
            
            logger.info(f"Model - MSE: {mse:.4f}, R2: {r2:.4f}, MAE: {mae:.4f}")
        
        # Extract feature importance
        self._extract_feature_importance(X_train)
        
        self.is_trained = True
        self.training_metrics = metrics
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict compatibility scores for candidate-job pairs.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted compatibility scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if self.model_type == 'ensemble':
            predictions = []
            
            for name, model in self.models.items():
                if name == 'neural_network':
                    X_scaled = self.scaler.transform(X)
                    pred = model.predict(X_scaled)
                else:
                    pred = model.predict(X)
                predictions.append(pred)
            
            # Return ensemble average
            return np.mean(predictions, axis=0)
        else:
            model = self.models['main']
            if self.model_type == 'neural_network':
                X_scaled = self.scaler.transform(X)
                return model.predict(X_scaled)
            else:
                return model.predict(X)
    
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict compatibility scores with confidence intervals.
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = self.predict(X)
        
        if self.model_type == 'ensemble':
            # Calculate confidence based on model agreement
            individual_predictions = []
            
            for name, model in self.models.items():
                if name == 'neural_network':
                    X_scaled = self.scaler.transform(X)
                    pred = model.predict(X_scaled)
                else:
                    pred = model.predict(X)
                individual_predictions.append(pred)
            
            # Confidence based on standard deviation of predictions
            prediction_std = np.std(individual_predictions, axis=0)
            confidence = 1.0 / (1.0 + prediction_std)  # Higher std = lower confidence
            
        else:
            # For single models, use a simple confidence measure
            # This could be improved with more sophisticated methods
            confidence = np.ones_like(predictions) * 0.8
        
        return predictions, confidence
    
    def get_feature_importance(self, feature_names: List[str] = None) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.feature_importance is None:
            return {}
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(self.feature_importance))]
        
        return dict(zip(feature_names, self.feature_importance))
    
    def save_model(self, filepath: str):
        """Save the trained model to disk."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_importance': self.feature_importance,
            'training_metrics': self.training_metrics,
            'is_trained': self.is_trained,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.feature_importance = model_data.get('feature_importance')
        self.training_metrics = model_data.get('training_metrics', {})
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Model loaded from {filepath}")
    
    def _optimize_hyperparameters(self, model, X_train: np.ndarray, y_train: np.ndarray):
        """Optimize hyperparameters using grid search."""
        logger.info("Optimizing hyperparameters...")
        
        if isinstance(model, RandomForestRegressor):
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif isinstance(model, GradientBoostingRegressor):
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 6, 9],
                'subsample': [0.8, 0.9, 1.0]
            }
        else:
            return model  # No optimization for other models
        
        grid_search = GridSearchCV(
            model, param_grid, cv=3, scoring='neg_mean_squared_error',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {-grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def _extract_feature_importance(self, X_train: np.ndarray):
        """Extract feature importance from trained models."""
        if self.model_type == 'ensemble':
            # Average importance across tree-based models
            importances = []
            
            for name, model in self.models.items():
                if hasattr(model, 'feature_importances_'):
                    importances.append(model.feature_importances_)
            
            if importances:
                self.feature_importance = np.mean(importances, axis=0)
            else:
                self.feature_importance = None
        else:
            model = self.models['main']
            if hasattr(model, 'feature_importances_'):
                self.feature_importance = model.feature_importances_
            else:
                self.feature_importance = None
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test feature matrix
            y_test: Test target values
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        predictions = self.predict(X_test)
        
        metrics = {
            'test_mse': mean_squared_error(y_test, predictions),
            'test_r2': r2_score(y_test, predictions),
            'test_mae': mean_absolute_error(y_test, predictions),
            'test_rmse': np.sqrt(mean_squared_error(y_test, predictions))
        }
        
        # Calculate accuracy within different thresholds
        for threshold in [0.1, 0.2, 0.3]:
            accuracy = np.mean(np.abs(predictions - y_test) <= threshold)
            metrics[f'accuracy_within_{threshold}'] = accuracy
        
        return metrics 