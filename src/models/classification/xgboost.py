from xgboost import XGBClassifier

from src.models.base_model import BaseModel


class XGBoostModel(BaseModel):

    def __init__(
        self,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=88,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    ):

        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            objective=objective,
            eval_metric=eval_metric,
            random_state=random_state,
            n_jobs=n_jobs,
        )

        self.model = XGBClassifier(**self.params)

    def fit(self, X_train, y_train, X_val=None, y_val=None):

        eval_set = None
    
        if X_val is not None:
            eval_set = [
                (X_train, y_train),
                (X_val, y_val),
            ]
    
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )
    
        if eval_set is not None:
            self.evals_result = self.model.evals_result()

    def predict(self, X):
        return self.model.predict(X)