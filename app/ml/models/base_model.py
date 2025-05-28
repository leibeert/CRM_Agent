from abc import ABC, abstractmethod
from typing import Any, Dict, List
import numpy as np
from sklearn.base import BaseEstimator

class BaseMLModel(ABC):
    """Base class for all ML models in the system."""
    
    def __init__(self):
        self.model: BaseEstimator = None
        self.is_trained: bool = False
    
    @abstractmethod
    def train(self, data: Any) -> None:
        """Train the model on the provided data."""
        pass
    
    @abstractmethod
    def predict(self, input_data: Any) -> Any:
        """Make predictions using the trained model."""
        pass
    
    @abstractmethod
    def evaluate(self, test_data: Any) -> Dict[str, float]:
        """Evaluate the model's performance."""
        pass
    
    def save(self, path: str) -> None:
        """Save the model to disk."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        # Implementation will be added in specific model classes
    
    def load(self, path: str) -> None:
        """Load the model from disk."""
        # Implementation will be added in specific model classes
        pass 