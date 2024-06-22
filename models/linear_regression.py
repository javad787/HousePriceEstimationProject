
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline


class LinearRegressionModel:
    """
    A class to encapsulate a Linear Regression model with preprocessing.

    Attributes:
    preprocessing (object): A preprocessing object that implements fit_transform and transform methods.
    model (object): The trained LinearRegression model. Initially set to None.
    """

    def __init__(self, preprocessing):
        """
        Initialize a LinearRegressionModel instance.

        Parameters:
        preprocessing (object): A preprocessing object that implements fit_transform and transform methods.
            This object is used to preprocess the input data before training and making predictions.
        """
        self.preprocessing = preprocessing
        self.model = None

    def train(self, X_train, y_train):
        """
        Train the Linear Regression model using the provided training data.

        Parameters:
        X_train (array-like): The input training data.
        y_train (array-like): The target training data.
        """
        lin_reg = make_pipeline(self.preprocessing, LinearRegression())
        lin_reg.fit(X_train, y_train)
        self.model = lin_reg

    def evaluate(self, X_test, y_test):
        """
        Evaluate the trained Linear Regression model using the provided test data.

        Parameters:
        X_test (array-like): The input test data.
        y_test (array-like): The target test data.

        Prints:
        The Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) of the model predictions.
        """
        print("Linear Regression MAE: ", mean_absolute_error(
            y_test, self.model.predict(X_test)))
        print("Linear Regression RMSE: ", mean_squared_error(
            y_test, self.model.predict(X_test), squared=False))