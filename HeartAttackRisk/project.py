import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

heart_risk_data = pd.read_csv("Heart Attack Data Set.csv") #first we need to read and implement the data set into the project
df = heart_risk_data.copy() #keeping the original data is better an we will apply the operations on this copy data set
"""Printing the first 5 row of data set"""
print(df.head())

"""With this code line we could see basic statistic about our data set and get some ideas about distribution of variables"""
print(df.describe().T) 

"""General information about dataset to observe data type or detect are there any missing values or not"""
print(df.info())

"""Detecting the number of unique values of each column and seperating the features as Categorcal and Numerical"""
for i in list(df.columns):
    print("{} -- {}".format(i, df[i].value_counts().shape[0]))

"""Making a corralation analysis would help us to understand the relation between 
features before build our model"""
plt.figure(figsize= (14, 10))
sns.heatmap(df.corr(), annot= True, fmt= ".2f", linewidths= .7)
plt.show()

"""Additional numeric feature analysis with target column"""
analysis_list = ["age", "trestbps", "chol", "thalach", "oldpeak", "target"]
df_analysis = df.loc[:, analysis_list]
sns.pairplot(data= df_analysis, hue= "target", diag_kind= "kde")
plt.show()

"""Outlers could distrupt ML process so they'd better being removed from data set
IQR = Q3 - Q1
Lower Bound = Q1 - (1.5 * IQR)
Upper Bound = Q3 + (1.5 * IQR)"""
numeric_list = ["age", "trestbps", "chol", "thalach", "oldpeak"]
for i in numeric_list:
    """In this loop we will find Q3, Q1 and IQR and after that we will calculate lower and upper bound with using them.
    Finally we need to remove the values that exceed those limits"""
    Q1 = np.percentile(df.loc[:, i], 25)
    Q3 = np.percentile(df.loc[:, i], 75)
    IQR = Q3 - Q1
    print("Old shape: ", df.loc[:, i].shape)
    #Upper Bound
    upper = np.where(df.loc[:, i] >= (Q3 + 1.5 * IQR))
    #Lower Bound
    lower = np.where(df.loc[:, i] <= (Q1 - 1.5 * IQR))
    print("{} -- {}".format(upper, lower))
    try:
        df.drop(upper[0], inplace= True)
    except: print("KeyError: {} not found in axis".format(upper[0]))

    try:
        df.drop(lower[0], inplace= True)
    except: print("KeyError: {} not found in axis".format(lower[0]))

    print("New shape: ", df.shape)

"""The last step before build a model is handling categorical datas and we can do this by using help of One Hot Encoding"""
df1 = df.copy()
categorical_list = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal", "target"]
df1 = pd.get_dummies(df1, columns= categorical_list[:-1], drop_first= True, dtype= float)
print(df1.head())

X = df1.drop(["target"], axis= 1)
y = df1[["target"]]

"The range between the values of each variable in the dataset may be different. We must apply standardization to the dataset"
scaler = StandardScaler()
X[numeric_list] = scaler.fit_transform(X[numeric_list])
print(X[numeric_list].head())

"""Applying train/test split"""
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state= 42)

def log_reg():
    """In this function we will build a logistic regression model, apply it on our data set and test it's success"""
    log = LogisticRegression()
    #fitting
    log_model = log.fit(x_train, y_train)
    #prediction
    y_pred = log_model.predict(x_test)
    log_score = accuracy_score(y_pred, y_test)
    return print("Test accuracy of Logistic Model => {}".format(log_score))

def svc():
    """In this function we will build a support vector machine model, apply it on our data set and test it's success"""
    svc = SVC(kernel= 'rbf')
    #fitting
    svc_model = svc.fit(x_train, y_train)
    #prediction
    y_pred = svc_model.predict(x_test)
    svc_score = accuracy_score(y_pred, y_test)
    return print("Test accuracy of Support Vector Machine Model => {}".format(svc_score))

def light():
    """In this function we will build a Light GBM model, apply it on our data set and test it's success"""
    lgbm = LGBMClassifier()
    #fitting
    lgbm_model = lgbm.fit(x_train, y_train)
    #prediction
    y_pred = lgbm_model.predict(x_test)
    lgbm_score = accuracy_score(y_pred, y_test)
    return print("Test accuracy of Light GBM Model => {}".format(lgbm_score))

light()
log_reg()
svc()

"""We have tested and seen that the Support Vector Machine model gave the highest accuracy score 
so it is our most reliable ML model. Now we can try to tune it and see if it returns with higher accuracy score"""

svc_model = SVC(kernel= 'rbf')
svc_model.fit(x_train, y_train)
y_pred1 = svc_model.predict(x_test)
accuracy_score1 = accuracy_score(y_pred1, y_test)

svc_params = {'C': [0.0001, 0.001, 0.1, 1, 5, 10,  50, 100],
              'gamma': [0.0001, 0.001, 0.1, 1, 5, 10, 50, 100],
              'cache_size': [50, 100, 200, 500, 1000]
              }

svc_cv_model = GridSearchCV(svc_model, svc_params, cv= 10, n_jobs= -1, verbose= 0)
svc_cv_model.fit(x_train, y_train)
print("Best parameters: " + str(svc_cv_model.best_params_))

svc_tuned = SVC(C= 100, cache_size= 50, gamma= 0.001).fit(x_train, y_train)
y_pred2 = svc_tuned.predict(x_test)
accuracy_score2 = accuracy_score(y_pred2, y_test)
print("\nFirst accuracy score => {} and tuned accuracy score => {}".format(accuracy_score1, accuracy_score2))