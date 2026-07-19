from sklearn.linear_model import LogisticRegression
from src.models.base_model import BaseModel

class LogisticRegressionModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = LogisticRegression(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    @property
    def feature_importances_(self):
        return self.model.coef_[0]
    
    @property
    def coefficients_(self):
        return self.model.coef_[0]
    
    @property
    def intercept_(self):
        return self.model.intercept_[0]