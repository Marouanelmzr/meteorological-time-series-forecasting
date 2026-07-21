from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.models.base_model import BaseModel

class LogisticRegressionModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(**kwargs)),
        ])

    def fit(self, X, y, X_val=None, y_val=None):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    @property
    def feature_importances_(self):
        return self.model.named_steps["classifier"].coef_[0]

    @property
    def coefficients_(self):
        return self.model.named_steps["classifier"].coef_[0]

    @property
    def intercept_(self):
        return self.model.named_steps["classifier"].intercept_[0]