
# Imports

# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation
import xgboost as xgb

# get bnp & test csv files as a DataFrame
bnp_df   = pd.read_csv("C://Users//Robert//Downloads//train.csv~//train.csv")
test_df  = pd.read_csv("C://Users//Robert//Downloads//test.csv~//test.csv")

# preview the data
bnp_df.head()

for f in bnp_df.columns:
    # fill NaN values with mean
    if bnp_df[f].dtype == 'float64':
        bnp_df.loc[:,f][np.isnan(bnp_df[f])] = bnp_df[f].mean()
        test_df[f][np.isnan(test_df[f])] = test_df[f].mean()
        
    # fill NaN values with most occured value
    elif bnp_df[f].dtype == 'object':
        bnp_df[f][bnp_df[f] != bnp_df[f]] = bnp_df[f].value_counts().index[0]
        test_df[f][test_df[f] != test_df[f]] = test_df[f].value_counts().index[0]
		
		from sklearn import preprocessing

for f in bnp_df.columns:
    if bnp_df[f].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(np.unique(list(bnp_df[f].values)  + list(test_df[f].values)))
        bnp_df[f]   = lbl.transform(list(bnp_df[f].values))
        test_df[f]  = lbl.transform(list(test_df[f].values))
		
# get Coefficient of Determination(R^2) for each feature using Logistic Regression
coeff_df = DataFrame(bnp_df.columns.delete([0,1]))
coeff_df.columns = ['Features']
coeff_df["Coefficient Estimate"] = (pd.Series(logreg.coef_[0])) ** 2

# preview
coeff_df.head()
# Plot coefficient of determination in order

coeff_ser = Series(list(coeff_df["Coefficient Estimate"]), index=coeff_df["Features"]).sort_values()
fig = coeff_ser.plot(kind='barh', figsize=(20,5))
fig.set(ylim=(100, 131))

#look at most significant param
x=bnp_df.loc[:,'v50'][bnp_df['target']==0]
y=bnp_df.loc[:,'v50'][bnp_df['target']==1]
data=[x,y]
plt.boxplot(data,0, 'D',labels=(0,1))
plt.show()
# not much to see
		

#random forest attempt all data
X_train = bnp_df.drop(["ID","target"],axis=1)
Y_train = bnp_df["target"]
X_test  = test_df.drop("ID",axis=1).copy()
# Random Forests
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict_proba(X_test)

#further visualization
#coeff_ser = Series(list(coeff_df["Coefficient Estimate"]), index=coeff_df["Features"]).sort_values()
#fig = coeff_ser.plot(kind='barh', figsize=(20,5))
#fig.set(ylim=(100, 131))
#fig.set(xlim=(0,0.01))

#random forest based on top significant columns  
#score was worse than using all data. (0.499)  apparently random forests do not work like linear regressions
#tt=X_test.iloc[:,(49,65,11,30,128,71,9,61,37,46,113,23,109,13,39)]
#tst=X_train.iloc[:,(49,65,11,30,128,71,9,61,37,46,113,23,109,13,39)]
#from sklearn.ensemble import RandomForestClassifier
#random_forest = RandomForestClassifier(n_estimators=100)
#random_forest.fit(tst, Y_train)
#Y_pred = random_forest.predict_proba(tt)

# Create submission

submission = pd.DataFrame()  
submission["ID"]            = test_df["ID"]
submission["PredictedProb"] = Y_pred[:,1]
submission.to_csv('bnp.csv', index=False)  #score .47423 --800th place =(