from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline


class RandomForestModel:
    """
    A class to encapsulate a Random Forest model with preprocessing.

    Attributes
    ----------
    preprocessing : sklearn.pipeline.Pipeline
        The preprocessing pipeline to apply before training the model.
    model : sklearn.ensemble.RandomForestRegressor
        The trained Random Forest model.

    Methods
    -------
    train(X_train, y_train)
        Trains the Random Forest model using the provided training data.

    evaluate(X_test, y_test)
        Evaluates the trained Random Forest model using the provided test data.
    """

    def __init__(self, preprocessing):
        """
        Constructs all the necessary attributes for the RandomForestModel object.

        Parameters
        ----------
        preprocessing : sklearn.pipeline.Pipeline
            The preprocessing pipeline to apply before training the model.
        """
        self.preprocessing = preprocessing
        self.model = None

    def train(self, X_train, y_train):
        """
        Trains the Random Forest model using the provided training data.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            The training input samples.
        y_train : array-like of shape (n_samples,)
            The target values (class labels) as integers or floats.
        """
        rf = make_pipeline(self.preprocessing, RandomForestRegressor(n_estimators=500, min_samples_split=5,
                                                                     min_samples_leaf=4, max_features=0.7, max_depth=10, bootstrap=False))
        rf.fit(X_train, y_train)
        self.model = rf

    def evaluate(self, X_test, y_test):
        """
        Evaluates the trained Random Forest model using the provided test data.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            The test input samples.
        y_test : array-like of shape (n_samples,)
            The target values (class labels) as integers or floats.

        Returns
        -------
        None
            Prints the Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) of the model predictions.
        """
        print("RF MAE: ", mean_absolute_error(
            y_test, self.model.predict(X_test)))
        print("RF RMSE: ", mean_squared_error(
            y_test, self.model.predict(X_test), squared=False))