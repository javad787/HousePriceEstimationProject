from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline


class LightGBMModel:
    """
    A class to encapsulate a LightGBM model with preprocessing.

    ...

    Attributes
    ----------
    preprocessing : sklearn.pipeline.Pipeline
        A preprocessing pipeline to be applied before training the model.
    model : sklearn.pipeline.Pipeline
        The trained LightGBM model.

    Methods
    -------
    train(X_train, y_train):
        Trains the LightGBM model using the provided training data.

    evaluate(X_test, y_test):
        Evaluates the trained model using the provided test data and prints the MAE and RMSE.
    """

    def __init__(self, preprocessing):
        """
        Constructs all the necessary attributes for the LightGBMModel class.

        Parameters
        ----------
        preprocessing : sklearn.pipeline.Pipeline
            A preprocessing pipeline to be applied before training the model.
        """
        self.preprocessing = preprocessing
        self.model = None

    def train(self, X_train, y_train):
        """
        Trains the LightGBM model using the provided training data.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            The training input samples.
        y_train : array-like of shape (n_samples,)
            The target values (class labels) as integers or floats.
        """
        light = make_pipeline(self.preprocessing, LGBMRegressor(
            boosting_type='gbdt',
            num_leaves=10,
            learning_rate=0.05, n_estimators=100, verbose=-1))

        light.fit(X_train, y_train)
        self.model = light

    def evaluate(self, X_test, y_test):
        """
        Evaluates the trained model using the provided test data and prints the MAE and RMSE.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            The test input samples.
        y_test : array-like of shape (n_samples,)
            The target values (class labels) as integers or floats.
        """
        print("LightGBM MAE: ", mean_absolute_error(
            y_test, self.model.predict(X_test)))
        print("LightGBM RMSE: ", mean_squared_error(
            y_test, self.model.predict(X_test), squared=False))