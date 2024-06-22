from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from skopt import BayesSearchCV


class KNNModel:
    """
    A class to encapsulate a K-Nearest Neighbors (KNN) model with preprocessing and hyperparameter tuning using BayesSearchCV.

    Attributes:
    preprocessing (object): A preprocessing object that implements fit_transform and transform methods.
    model (object): The trained KNN model. Initially set to None.
    """

    def __init__(self, preprocessing):
        """
        Initialize a KNNModel instance.

        Parameters:
        preprocessing (object): A preprocessing object that implements fit_transform and transform methods.
            This object is used to preprocess the input data before training and making predictions.
        """
        self.preprocessing = preprocessing
        self.model = None

    def train(self, X_train, y_train):
        """
        Train the KNN model using the provided training data and perform hyperparameter tuning using BayesSearchCV.

        Parameters:
        X_train (array-like): The input training data.
        y_train (array-like): The target training data.
        """
        knnreg = KNeighborsRegressor()
        param_space = {
            'n_neighbors': (1, 10),
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': (10, 50),
            'p': (1, 2)
        }

        knn = make_pipeline(self.preprocessing, BayesSearchCV(
            knnreg, param_space, n_iter=1, scoring='neg_mean_squared_error', cv=5, verbose=False))

        knn.fit(X_train, y_train)
        self.model = knn

    def evaluate(self, X_test, y_test):
        """
        Evaluate the trained KNN model using the provided test data.

        Parameters:
        X_test (array-like): The input test data.
        y_test (array-like): The target test data.

        Prints:
        The Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) of the model predictions.
        """
        print("KNN MAE: ", mean_absolute_error(
            y_test, self.model.predict(X_test)))
        print("KNN RMSE: ", mean_squared_error(
            y_test, self.model.predict(X_test), squared=False))