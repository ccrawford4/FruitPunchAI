# Author: Calum Crawford
# Computer Science Student at the University of San Francisco

# Training and Testing Different Machine Learning models

from sklearn.datasets import load_iris, load_diabetes, load_digits, load_wine, load_breast_cancer
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# Put all the data sets in a list to easily parse through them and evaluate
dataSets = [load_iris, load_diabetes, load_digits, load_wine, load_breast_cancer] 

for i in range(len(dataSets)):
  scores =[]
  module = []
  # Conducts the DecisionTree algorithm for the data sets
  X = dataSets[i]().data
  y = dataSets[i]().target

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

  tree = DecisionTreeClassifier()
  fitted_tree = tree.fit(X_train, y_train)
  y_pred = fitted_tree.predict(X_test)

  acc_score = accuracy_score(y_test, y_pred)
  r2 = r2_score(y_test, y_pred)

  scores.append(acc_score)
  module.append(DecisionTreeClassifier.__name__)

  print(f'On the {dataSets[i].__name__} dataset the {DecisionTreeClassifier.__name__} reaches an accuracy score of {acc_score} and a R^2 score of {r2}')


 # Conducts the Neural Network algorithm for the data sets

  if dataSets[i].__name__ == "load_iris":
    mlp = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', solver='sgd', max_iter=500, random_state=0, early_stopping=True)
  else:
    mlp = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', solver='sgd', max_iter=500, random_state=0)

  mlp.fit(X_train, y_train)
  y_pred = mlp.predict(X_test)
  r2 = r2_score(y_test, y_pred)
  acc_score = accuracy_score(y_test, y_pred)

  print(f'On the {dataSets[i].__name__} dataset the {MLPClassifier.__name__} reaches an accuracy score of {acc_score} and a R^2 score of {r2}')

  scores.append(acc_score)
  module.append(MLPClassifier.__name__)

    # Conducts the Neighbors algorithm for the data sets
  knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  knn.fit(X_train, y_train)

  y_pred = knn.predict(X_test)
  r2 = r2_score(y_test, y_pred)
  acc_score = accuracy_score(y_test, y_pred)

  print(f'On the {dataSets[i].__name__} dataset the {KNeighborsClassifier.__name__} reaches an accuracy score of {acc_score} and a R^2 score of {r2}')

  scores.append(acc_score)
  module.append(KNeighborsClassifier.__name__)

 # Conducts the Ensemble algorithm for the data set
  X_train, X_test, y_train, y_test, = train_test_split(X, y, test_size=0.2, random_state=42)
  rf = RandomForestClassifier(n_estimators=100, random_state=42)
  rf.fit(X_train, y_train)

  y_pred = rf.predict(X_test)
  r2 = r2_score(y_test, y_pred)
  acc_score = accuracy_score(y_test, y_pred)

  print(f'On the {dataSets[i].__name__} dataset the {RandomForestClassifier.__name__} reaches an accuracy score of {acc_score} and a R^2 score of {r2}')

  scores.append(acc_score)
  module.append(RandomForestClassifier.__name__)

 # Conducts the Naive Byes algorithm for the data set
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  nb = GaussianNB()
  nb.fit(X_train, y_train)

  y_pred = nb.predict(X_test)
  r2 = r2_score(y_test, y_pred)
  acc_score = accuracy_score(y_test, y_pred)

  print(f'On the {dataSets[i].__name__} dataset the {GaussianNB.__name__} reaches an accuracy score of {acc_score} and a R^2 score of {r2}')

  scores.append(acc_score)
  module.append(GaussianNB.__name__)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Concucts a Logistic Regression evaluation
  lr = LogisticRegression(max_iter=10000)
  lr.fit(X_train, y_train)

  y_pred = lr.predict(X_test)
  r2 = r2_score(y_test, y_pred)
  acc_score = accuracy_score(y_test, y_pred)

  print(f'On the {dataSets[i].__name__} dataset the {LogisticRegression.__name__} reaches an accuracy score of {acc_score} and a R^2 score of {r2}')

  scores.append(acc_score)
  module.append(LogisticRegression.__name__)
  
  topScore = min(scores, key=lambda x: abs(x-1.0))
  index = scores.index(topScore)
  topModule = module[index]

  print()
  print(f'The {topModule} model worked best for the {dataSets[i].__name__} dataset with a accuracy score of {topScore}')
  print()