from sklearn.tree import DecisionTreeRegressor

from ngboost import NGBRegressor
from ngboost.distns import LogNormal, Gamma
from ngboost.scores import LogScore

from src.models.base_model import BaseModel


class NGBoostModel(BaseModel):

    DISTRIBUTIONS = {
        "lognormal": LogNormal,
        "gamma": Gamma,
    }

    def __init__(
        self,
        distribution="lognormal",
        n_estimators=500,
        learning_rate=0.05,
        base_max_depth=3,
        base_min_samples_leaf=20,
        base_min_samples_split=10,
        minibatch_frac=1.0,
        col_sample=1.0,
        natural_gradient=True,
        random_state=42,
        verbose=False,
        early_stopping_rounds=None,
    ):

        super().__init__(
            distribution=distribution,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            base_max_depth=base_max_depth,
            base_min_samples_leaf=base_min_samples_leaf,
            base_min_samples_split=base_min_samples_split,
            minibatch_frac=minibatch_frac,
            col_sample=col_sample,
            natural_gradient=natural_gradient,
            random_state=random_state,
            verbose=verbose,
            early_stopping_rounds=early_stopping_rounds,
        )

        distribution = distribution.lower()

        if distribution not in self.DISTRIBUTIONS:
            raise ValueError(
                f"Unknown distribution '{distribution}'. "
                f"Available: {list(self.DISTRIBUTIONS.keys())}"
            )

        self.model = NGBRegressor(

            Dist=self.DISTRIBUTIONS[distribution],

            Score=LogScore,

            Base=DecisionTreeRegressor(
                max_depth=base_max_depth,
                min_samples_leaf=base_min_samples_leaf,
                min_samples_split=base_min_samples_split,
                random_state=random_state,
            ),

            n_estimators=n_estimators,
            learning_rate=learning_rate,
            minibatch_frac=minibatch_frac,
            col_sample=col_sample,
            natural_gradient=natural_gradient,
            random_state=random_state,
            verbose=verbose,
            early_stopping_rounds=early_stopping_rounds,
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None):

        if X_val is not None:

            self.model.fit(
                X_train,
                y_train,
                X_val=X_val,
                Y_val=y_val,
            )

        else:

            self.model.fit(
                X_train,
                y_train,
            )

    def predict(self, X):

        return self.predict_distribution(X).mean()

    def predict_distribution(self, X):

        return self.model.pred_dist(X)

    def predict_mean(self, X):

        return self.predict_distribution(X).mean()

    def predict_median(self, X):

        return self.predict_distribution(X).median()

    def predict_std(self, X):

        return self.predict_distribution(X).std()

    def predict_quantile(self, X, q):

        return self.predict_distribution(X).ppf(q)

    def predict_proba(self, X):

        raise NotImplementedError(
            "Regression models do not support predict_proba()."
        )