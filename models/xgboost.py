from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
from skopt import BayesSearchCV


class XGBoostModel:
    """
    A class to encapsulate the functionality of an XGBoost model with hyperparameter tuning.

    Attributes
    ----------
    preprocessing : sklearn.pipeline.Pipeline
        A preprocessing pipeline for the input data.
    model : sklearn.pipeline.Pipeline
        The trained XGBoost model with hyperparameter tuning.

    Methods
    -------
    train(X_train, y_train)
        Trains the XGBoost model with hyperparameter tuning using the given training data.

    predict(X_test)
        Predicts the target values for the given test data using the trained model.

    evaluate(X_test, y_test)
        Evaluates the trained model using mean absolute error and root mean squared error metrics.
    """

    def __init__(self, preprocessing):
        """
        Constructs all the necessary attributes for the XGBoostModel object.

        Parameters
        ----------
        preprocessing : sklearn.pipeline.Pipeline
            A preprocessing pipeline for the input data.
        """
        self.preprocessing = preprocessing
        self.model = None

    def train(self, X_train, y_train):
        """
        Trains the XGBoost model with hyperparameter tuning using the given training data.

        Parameters
        ----------
        X_train : array-like
            The input training data.
        y_train : array-like
            The target values for the training data.
        """
        regressor = XGBRegressor()

        param_space = {
            'learning_rate': (0.01, 0.3, 'log-uniform'),
            'max_depth': (3, 10),
            'n_estimators': (100, 1000),
            'subsample': (0.5, 1.0, 'uniform'),
            'colsample_bytree': (0.5, 1.0, 'uniform'),
            'colsample_bylevel': (0.5, 1.0, 'uniform'),
            'reg_lambda': (0.0, 1.0, 'uniform'),
            'reg_alpha': (0.0, 1.0, 'uniform'),
            'min_child_weight': (1, 10)
        }

        xgb = make_pipeline(self.preprocessing, BayesSearchCV(
            regressor, param_space, n_iter=1, scoring='neg_mean_squared_error', cv=5, verbose=False))
        xgb.fit(X_train, y_train)
        self.model = xgb

    def predict(self, X_test):
        """
        Predicts the target values for the given test data using the trained model.

        Parameters
        ----------
        X_test : array-like
            The input test data.

        Returns
        -------
        array-like
            The predicted target values for the test data.
        """
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        """
        Evaluates the trained model using mean absolute error and root mean squared error metrics.

        Parameters
        ----------
        X_test : array-like
            The input test data.
        y_test : array-like
            The target values for the test data.
        """
        print("XGBoost MAE: ", mean_absolute_error(
            y_test, self.model.predict(X_test)))
        print("XGBoost RMSE: ", mean_squared_error(
            y_test, self.model.predict(X_test), squared=False))