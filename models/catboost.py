from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from skopt import BayesSearchCV


class CatBoostModel:
    """
    A class to encapsulate a CatBoost model with preprocessing and hyperparameter tuning using BayesSearchCV.

    Attributes:
    preprocessing (object): A preprocessing object that implements fit_transform and transform methods.
    model (object): The trained CatBoost model. Initially set to None.
    """

    def __init__(self, preprocessing):
        """
        Initialize a CatBoostModel instance.

        Parameters:
        preprocessing (object): A preprocessing object that implements fit_transform and transform methods.
            This object is used to preprocess the input data before training and making predictions.
        """
        self.preprocessing = preprocessing
        self.model = None

    def train(self, X_train, y_train):
        """
        Train the CatBoost model using the provided training data and perform hyperparameter tuning using BayesSearchCV.

        Parameters:
        X_train (array-like): The input training data.
        y_train (array-like): The target training data.
        """
        catboost = CatBoostRegressor(verbose=False)

        param_space = {
            'learning_rate': (0.01, 0.3, 'log-uniform'),
            'depth': (3, 10),
            'n_estimators': (100, 1000),
            'l2_leaf_reg': (1, 10),
            'subsample': (0.5, 1.0, 'uniform'),
            'colsample_bylevel': (0.5, 1.0, 'uniform'),
            'min_child_samples': (1, 20),
            'border_count': (1, 255)
        }

        cat_b = make_pipeline(self.preprocessing, BayesSearchCV(
            catboost, param_space, n_iter=5, scoring='neg_mean_squared_error', cv=5, verbose=False))

        cat_b.fit(X_train, y_train)
        self.model = cat_b

    def evaluate(self, X_test, y_test):
        """
        Evaluate the trained CatBoost model using the provided test data.

        Parameters:
        X_test (array-like): The input test data.
        y_test (array-like): The target test data.

        Prints:
        The Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) of the model predictions.
        """
        print("Catboost MAE: ", mean_absolute_error(
            y_test, self.model.predict(X_test)))
        print("Catboost RMSE: ", mean_squared_error(
            y_test, self.model.predict(X_test), squared=False))