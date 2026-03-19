from sklearn.linear_model import Ridge
import numpy as np
from sklearn.model_selection import train_test_split

# 1. Create synthetic data
X, y = np.random.randn(10, 5), np.random.randn(10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)





def ridgeRegression(X,y,weights):

    model = Ridge(alpha=1.0)

    model.fit(X*weights, y)

    return model

w = [1,0.2,3,5,6]
m = ridgeRegression(X_train, y_train, w)
print("Model coefs: ", m.coef_)
print("Model intercepts: ", m.intercept_)
predictions = m.predict(X_test*w)
print("Predictions: ", predictions)