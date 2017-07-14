import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold
from re import sub
from decimal import Decimal
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

fraudInstanceData = pd.read_csv("FraudInstanceData.csv", header=0, index_col=0)
maritalStatuses = pd.get_dummies(fraudInstanceData["Marital Status"])
accomodationTypes = pd.get_dummies(fraudInstanceData["Accomodation Type"])
fraudInstanceData = fraudInstanceData.drop('Marital Status', axis=1)
fraudInstanceData = fraudInstanceData.drop('Accomodation Type', axis=1)
fraudInstanceData = fraudInstanceData.join(maritalStatuses)
fraudInstanceData = fraudInstanceData.join(accomodationTypes)

currencyToMoney = lambda c: Decimal(sub(r'[^\d.]', '', c))
fraudInstanceData['Claim Amount'] = fraudInstanceData["Claim Amount"].apply(currencyToMoney)

y = fraudInstanceData.iloc[:, 1]
X = fraudInstanceData.iloc[:, 1:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=23)

pipeline = Pipeline([('feature_selection', SelectFromModel(LogisticRegression(penalty="l1"))),
                     ('regression', LogisticRegression())])
grid_cv = GridSearchCV(pipeline, {}, cv=10)
grid_cv.fit(X_train, y_train)

selected_feature = grid_cv.transform(X_train.co)

y_pred = grid_cv.predict(X_test)
print(grid_cv.score(X_test, y_pred))
print(confusion_matrix(y_test, y_pred))
