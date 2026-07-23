from xgboost import XGBRegressor

from src.models.base_model import BaseModel


class XGBoostRegressorModel(BaseModel):

    def __init__(
        self,
        n_estimators=800,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0,
        reg_alpha=0,
        reg_lambda=1,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=None,
    ):

        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            objective=objective,
            eval_metric=eval_metric,
            random_state=random_state,
            n_jobs=n_jobs,
            early_stopping_rounds=early_stopping_rounds,
        )

        self.model = XGBRegressor(**self.params)

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

    def predict_proba(self, X):
        raise NotImplementedError(
            "Regression models do not support predict_proba()."
        )