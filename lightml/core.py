import pandas as pd
import numpy as np
import pickle

class SmartTransformer:    
    @staticmethod
    def transform(X, y):
        X = X.values
        y = y.values

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X, y

class Linear_Regression_OLS:
    def __init__(self):
        self.coef_ = None

    def fit(self, X_train, y_train):
        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)

        X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
        self.coef_ = np.dot(np.linalg.inv(np.dot(X_train.T, X_train)), np.dot(X_train.T, y_train))

    def predict(self, X_test):
        X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
        return np.dot(X_test, self.coef_)

    def returnScore(self, X_test, y_test):
        y_pred = self.predict(X_test)
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        return 1 - (ss_res / ss_tot)

    def getEquation(self):
        intercept = round(self.coef_[0], 5)
        coefs = np.round(self.coef_[1:], 5)
        return f"y_hat = {intercept} + {coefs} * X"

class Linear_regression_BGD:
    def __init__(self, epochs, learning_rate):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.coef_ = None

    def fit(self, X_train, y_train):
        if (X_train.ndim == 1):
            X_train = X_train.reshape(-1, 1)

        X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
        self.coef_ = np.zeros( X_train.shape[1])
        n = X_train.shape[0]

        for _ in range(self.epochs):
            error = y_train - np.dot(X_train, self.coef_)
            slope = (-2/n) * (np.dot(X_train.T, error))
            self.coef_ -= (self.learning_rate * slope) 

    def predict(self, X_test):
        X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
        return np.dot(X_test, self.coef_)

    def returnScore(self, X_test, y_test):
        y_pred = self.predict(X_test)
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        return 1 - (ss_res / ss_tot)

    def getEquation(self):
        intercept = round(self.coef_[0], 5)
        coefs = np.round(self.coef_[1:], 5)
        return f"y_hat = {intercept} + {coefs} * X"
    
    

class Linear_Regression_BGD_L1:
    def __init__(self, epochs, learning_rate, theLamda):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.coef_ = None
        self.thisLambda = theLamda

    def fit(self, X_train, y_train):
        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)
        if y_train.ndim > 1:
            y_train = y_train.ravel()

        X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
        self.coef_ = np.zeros(X_train.shape[1])
        n = X_train.shape[0]

        for _ in range(self.epochs):
            error = y_train - np.dot(X_train, self.coef_)
            slope = (-2 / n) * np.dot(X_train.T, error) + self.thisLambda * np.sign(self.coef_)
            self.coef_ -= self.learning_rate * slope
    def predict(self, X_test):
        X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
        return np.dot(X_test, self.coef_)

    def returnScore(self, X_test, y_test):
        y_pred = self.predict(X_test)
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        return 1 - (ss_res / ss_tot)

    def getEquation(self):
        intercept = round(self.coef_[0], 5)
        coefs = np.round(self.coef_[1:], 5)
        return f"y_hat = {intercept} + {coefs} * X"
        
class lightml:
    def __init__(self, df):
        self.dataframe = df.copy()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.target_column = None

    def fit_data(self, random_state=42, ratio=0.8, training_features=None, target_features=None):
        if target_features is None:
            raise ValueError("Target Feature parameter can't be None")
        elif len(target_features) > 1:
            raise ValueError("Target Feature can't be greater than 1")

        if training_features is None:
            training_features = self.dataframe.columns.drop(target_features[0])

        self.target_column = target_features[0]

        # Filter only selected features
        df_filtered = self.dataframe[training_features + [self.target_column]]

        # Train-test split
        train = df_filtered.sample(frac=ratio, random_state=random_state)
        test = df_filtered.drop(train.index)

        self.X_train = train[training_features]
        self.y_train = train[self.target_column]
        self.X_test = test[training_features]
        self.y_test = test[self.target_column]
        self.X_train, self.y_train = SmartTransformer.transform(self.X_train, self.y_train)
        self.X_test, self.y_test = SmartTransformer.transform(self.X_test, self.y_test)


    def Standard_Scale(self, features=None):
        def standardized(X):
            mean = X.mean(axis=0)
            std = X.std(axis=0)
            return (X - mean) / std

        if features is None:
            features = self.dataframe.columns.tolist()

        self.dataframe[features] = standardized(self.dataframe[features])


    def save_model(self, model_obj, filename="model.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(model_obj, f)

    def load_model(self, filename="model.pkl"):
        with open(filename, "rb") as f:
            return pickle.load(f)
    
    def doML(self, get_equation = False, model = None, method = None, penalty = None, epochs = 100, learning_rate = 0.01, lamda_ = 0.01):
        if (model is None or method is None):
            raise ValueError("Parameter can't be None")
        
        if (model == "lr"):

            if (method == "ols"):
                lr = Linear_Regression_OLS()
                lr.fit(self.X_train, self.y_train)
                if (get_equation):
                    print(lr.getEquation())
                return lr.returnScore(self.X_test, self.y_test)
            elif (method == "bgd" and penalty is None):
                lr = Linear_regression_BGD(epochs=epochs, learning_rate=learning_rate)
                lr.fit(self.X_train, self.y_train)
                if (get_equation):
                    print(lr.getEquation())
                return lr.returnScore(self.X_test, self.y_test)
            
            elif (method == 'bgd' and penalty == 'l1'):
                lr = Linear_Regression_BGD_L1(epochs=epochs, learning_rate=learning_rate, theLamda=lamda_)
                lr.fit(self.X_train, self.y_train)
                if (get_equation):
                    print(lr.getEquation())
                return lr.returnScore(self.X_test, self.y_test)
