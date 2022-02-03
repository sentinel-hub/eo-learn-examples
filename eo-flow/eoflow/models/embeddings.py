from sklearn.base import BaseEstimator,RegressorMixin
from sklearn.exceptions import NotFittedError

class EmbeddingEstimator(BaseEstimator,RegressorMixin):
    def __init__(self, model):
        self.model = model
        self.estimator_fitted = False

    def get_embeddings(self, x):
        self.model.predict(x)
        return self.model.get_feature_map(inputs=x)

    def define_estimator(self, estimator):
        self.estimator = estimator
        return self

    def fit(self, x,y):
        x = self.get_embeddings(x)
        self.estimator.fit(x, y)
        self.estimator_fitted = True
        return self

    def predict(self, x):
        if not self.estimator_fitted:
            raise NotFittedError(
                'The estimator must be fitted before calling predict(...).'
            )
        x = self.get_embeddings(x)
        return self.estimator.predict(x)