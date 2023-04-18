# Author: Calum Crawford
# Computer Science Student at the University of San Francisco

# Using XGBoost to predict given datasets

from sklearn.datasets import load_iris, load_diabetes, load_digits, load_wine, load_breast_cancer
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

dataSets = [load_iris, load_diabetes, load_digits, load_wine, load_breast_cancer]

for i in range(len(dataSets)):

  X = dataSets[i]().data
  y = dataSets[i]().target

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  model = XGBRegressor()

  model.fit(X_train, y_train)

  y_pred = model.predict(X_test)
  r2 = r2_score(y_test, y_pred)
# Evaluates the XGBRegressor model's r squared score for each of the data sets
  print(f'On the {dataSets[i].__name__} dataset {XGBRegressor.__name__} module reaches an R^2 score of {r2}')