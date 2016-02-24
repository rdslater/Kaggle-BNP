
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

data = pd.read_csv("C://Users//Robert//Downloads//train.csv~//train.csv")

result=data.iloc[:,1].as_matrix
v1_0.head(10)
x=v1_0.v1
y=v1_0.v2
col=v1_0.target
plt.scatter(x,y,c=col)


# In[26]:

v1_0=data.iloc[:,1:4].dropna()
v1_1=data["v1"][data.target==1].dropna()


# In[13]:


# Imports

# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().magic('matplotlib inline')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation
import xgboost as xgb


# In[14]:

# get bnp & test csv files as a DataFrame
bnp_df   = pd.read_csv("C://Users//Robert//Downloads//train.csv~//train.csv")
test_df  = pd.read_csv("C://Users//Robert//Downloads//test.csv~//test.csv")

# preview the data
bnp_df.head()


# In[15]:

for f in bnp_df.columns:
    # fill NaN values with mean
    if bnp_df[f].dtype == 'float64':
        bnp_df.loc[:,f][np.isnan(bnp_df[f])] = bnp_df[f].mean()
        test_df[f][np.isnan(test_df[f])] = test_df[f].mean()
        
    # fill NaN values with most occured value
    elif bnp_df[f].dtype == 'object':
        bnp_df[f][bnp_df[f] != bnp_df[f]] = bnp_df[f].value_counts().index[0]
        test_df[f][test_df[f] != test_df[f]] = test_df[f].value_counts().index[0]


# In[19]:

bnp_df.head()


# In[16]:

from sklearn import preprocessing

for f in bnp_df.columns:
    if bnp_df[f].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(np.unique(list(bnp_df[f].values)  + list(test_df[f].values)))
        bnp_df[f]   = lbl.transform(list(bnp_df[f].values))
        test_df[f]  = lbl.transform(list(test_df[f].values))


# In[10]:

bnp_df.head()


# In[17]:

X_train = bnp_df.drop(["ID","target"],axis=1)
Y_train = bnp_df["target"]
X_test  = test_df.drop("ID",axis=1).copy()


# In[24]:


# Random Forests
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict_proba(X_test)



# In[11]:


Y_pred = random_forest.predict_proba(X_test)


# In[23]:


# Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict_proba(X_test)[:,1]

logreg.score(X_train, Y_train)
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


# In[24]:

Y_pred


# In[62]:

x=bnp_df.loc[:,'v50'][bnp_df['target']==0]
y=bnp_df.loc[:,'v50'][bnp_df['target']==1]


# In[77]:

data=[x,y]
plt.boxplot(data,0, 'D',labels=(0,1))

plt.show()


# In[70]:

help(plt.boxplot)


# In[27]:

coeff_ser = Series(list(coeff_df["Coefficient Estimate"]), index=coeff_df["Features"]).sort_values()
fig = coeff_ser.plot(kind='barh', figsize=(20,5))
fig.set(ylim=(100, 131))
fig.set(xlim=(0,0.01))


# In[29]:

Y_pred[:,1]


# In[40]:

# Create submission

submission = pd.DataFrame()
submission["ID"]            = test_df["ID"]
submission["PredictedProb"] = Y_pred[:,1]

submission.to_csv('bnp.csv', index=False)


# In[39]:

tt=X_test.iloc[:,(49,65,11,30,128,71,9,61,37,46,113,23,109,13,39)]
tst=X_train.iloc[:,(49,65,11,30,128,71,9,61,37,46,113,23,109,13,39)]
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict_proba(X_test)


# In[41]:

Y_pred


# In[ ]:



