from abc import ABC, abstractmethod
from pathlib import Path

import joblib

class BaseModel(ABC):
    def __init__(self, **kwargs):
        self.params = kwargs
        self.model = None

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        
        raise NotImplementedError("This model does not support probability predictions.")

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
        return self
    
    def get_params(self):
        return self.params
    
    def set_params(self, **kwargs):
        self.params.update(kwargs)
        if self.model is not None:
            self.model.set_params(**kwargs)
        return self