# write your code here
"""
Load a pandas Dataframe containing x and y;
Initialize CustomLinearRegression class;
Implement the fit() method;
Initialize your linear regression class with fit_intercept=True;
Fit the provided data;
Print a dictionary containing the intercept and coefficient values.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

data = pd.DataFrame({
    'X': [4, 4.5, 5, 5.5, 6, 6.5, 7],
    "W": [1, -3, 2, 5, 0, 3, 6],
    "Z": [11, 15, 12, 9, 18, 13, 16],
    'y': [33, 42, 45, 51, 53, 61, 62]})

data1 = pd.DataFrame({
    "X": [1, 2, 3, 4, 10.5],
    "W": [7.5, 10, 11.6, 7.8, 13],
    "Z": [26.7, 6.6, 11.9, 72.5, 2.1],
    "y": [105.6, 210.5, 177.9, 154.7, 160]
})

data3 = pd.DataFrame({
    'Capacity': [0.9, 0.5, 1.75, 2.0, 1.4, 1.5, 3.0, 1.1, 2.6, 1.9],
    'Age': [11, 11, 9, 8, 7, 7, 6, 5, 5, 4],
    'Cost/ton': [21.95, 27.18, 16.9, 15.37, 16.03, 18.15, 14.22, 18.72, 15.4, 14.69],
})

data4 = pd.read_csv("data_stage4.csv")

X_train = data4[["f1", "f2", "f3"]]
y_train = data4["y"]


class CustomLinearRegression:

    def __init__(self, *args, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.intercept = 0
        self.coefficient = np.array(0)

    def fit(self, X, y):
        if self.fit_intercept:
            # creating an array with 1's
            intercept_array = np.ones((X.shape[0], 1))
            # adding intercept column to X
            Xintercepted = np.concatenate([intercept_array, X], axis=1)
            Xintercepted, y = np.array(Xintercepted), np.array(y)
            B = np.matmul(np.matmul(np.linalg.inv(np.matmul(Xintercepted.T, Xintercepted)), Xintercepted.T), y)
            self.intercept = np.float_(B[0])
            self.coefficient = B[1:]
        else:
            X, y = np.array(X), np.array(y)
            B = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), y)
            self.intercept = 0
            self.coefficient = np.array(B)

    def predict(self, X):
        if self.fit_intercept:
            X = np.array(X)
            intercept_array = np.ones((X.shape[0], 1))
            Xintercepted = np.concatenate([intercept_array, X], axis=1)
            y_predict = np.matmul(Xintercepted, [self.intercept, *self.coefficient])
            return y_predict
        else:
            X = np.array(X)
            y_predict = np.matmul(X, self.coefficient)
            return y_predict

    def r2_score(self, y_real, y_predict):
        sum1 = sum([(yr - yp) ** 2 for yr, yp in zip(y_real, y_predict)])
        sum2 = sum([(yr - np.mean(y_real)) ** 2 for yr, yp in zip(y_real, y_predict)])
        score = 1 - (sum1 / sum2)
        return score

    def rmse(self, y_real, y_predict):
        mse = sum([(yr - yp) ** 2 for yr, yp in zip(y_real, y_predict)]) / len(y_real)
        rmse = mse ** 0.5
        return rmse


model_c = CustomLinearRegression(fit_intercept=True)
model_c.fit(X_train, y_train)
predict_c = model_c.predict(X_train)
r2_c = model_c.r2_score(y_train, predict_c)
rmse_c = model_c.rmse(y_train, predict_c)

model = LinearRegression(fit_intercept=True)
model.fit(X_train, y_train)
predict = model.predict(X_train)
r2 = r2_score(y_train, predict)
rmse = mean_squared_error(y_train, predict, squared=False)

result = {
    'Intercept': model_c.intercept - model.intercept_,
    'Coefficient': model_c.coefficient - model.coef_,
    'R2': r2_c - r2,
    'RMSE': rmse_c - rmse}

print(result)
