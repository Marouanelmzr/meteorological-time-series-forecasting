from lightgbm import LGBMClassifier

from src.models.base_model import BaseModel

from lightgbm import record_evaluation


class LightGBMModel(BaseModel):

    def __init__(
        self,
        n_estimators=800,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=42,
        n_jobs=-1,
    ):

        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=n_jobs,
        )

        self.model = LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=n_jobs,
            objective="binary",
            class_weight="balanced",
            verbosity=-1,
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None):
    
        evals_result = {}
    
        if X_val is not None:
        
            self.model.fit(
                X_train,
                y_train,
                eval_set=[
                    (X_train, y_train),
                    (X_val, y_val),
                ],
                callbacks=[
                    record_evaluation(evals_result),
                ],
            )
    
            self.evals_result = evals_result
    
        else:
        
            self.model.fit(
                X_train,
                y_train,
            )

    def predict(self, X):
        return self.model.predict(X)