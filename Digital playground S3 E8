#!/usr/bin/env python
# coding: utf-8

# In[123]:



import pandas as pd 
import seaborn as sns
import numpy as np 
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, mean_squared_error
import xgboost as xgb
import lightgbm as lgbm
import catboost


# this is where ı get some information about columns
# A description of the clarity:
# 
# https://abovediamond.com/learn-diamond-clarity/
# 
# 
# Carat: a weight of the cubic zirconia. A metric “carat” is defined as 200 milligrams.
# Cut:  describes the cut quality of the cubic zirconia. Quality is increasing order Fair, Good, Very Good, Premium, Ideal.
# Color:  refers to the color of the cubic zirconia. With D being the best and J the worst.
# Clarity: refers to the absence of the Inclusions and Blemishes. (In order from Best to Worst, FL = flawless, I3= level 3 inclusions) FL, IF, VVS1, VVS2, VS1, VS2, SI1, SI2, I1, I2, I3
# Depth:  the height of a cubic zirconia, measured from the Culet to the table, divided by its average Girdle Diameter.
# Table:  the width of the cubic zirconia's Table expressed as a Percentage of its Average Diameter.
# X:  Length of the cubic zirconia in mm.
# Y:  Width of the cubic zirconia in mm.
# Z:  Height of the cubic zirconia in mm.

# In[89]:


train = pd.read_csv("train.csv").drop(columns = "id")


# In[90]:


origin = pd.read_csv("original_dataset.csv").drop(columns='Unnamed: 0')


# In[91]:


test = pd.read_csv("test.csv").drop(columns = "id")


# In[131]:


splgm = pd.read_csv("sample_submission.csv")
spens = pd.read_csv("sample_submission.csv")
spcat = pd.read_csv("sample_submission.csv")


# In[93]:


train.head(20)


# In[94]:


train.shape , test.shape


# In[95]:


train.info(),test.info()


# 

# In[96]:


train.describe().T


# In[97]:


target = "price"


# In[98]:


train[target].head(20)


# In[99]:


train.isnull().sum() ,test.isnull().sum(),origin.isnull().sum()


# TARGET DİSTRİBUTİON

# In[100]:


plt.figure(figsize=(8,6))
sns.kdeplot(data=train, x=target, label='train')
sns.kdeplot(data=origin, x=target, label='original')
plt.legend()


# In[101]:


from sklearn.preprocessing import OrdinalEncoder


# In[102]:


def process(df):
    df['cut'] = df['cut'].apply(lambda x: cut_dic[x])
    df['color'] = df['color'].apply(lambda x:color_dic[x])
    df['clarity'] = df['clarity'].apply(lambda x:clarity_dic[x])
    df["volume"] = df["x"] * df["y"] * df["z"]
    df["surface_area"] = 2 * (df["x"] * df["y"] + df["y"] * df["z"] + df["z"] * df["x"])
    df["aspect_ratio_xy"] = df["x"] / df["y"]
    df["aspect_ratio_yz"] = df["y"] / df["z"]
    df["aspect_ratio_zx"] = df["z"] / df["x"]
    df["diagonal_distance"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2)
    df["relative_height"] = (df["z"] - df["z"].min()) / (df["z"].max() - df["z"].min())
    df["relative_position"] = (df["x"] + df["y"] + df["z"]) / (df["x"] + df["y"] + df["z"]).sum()
    df["volume_ratio"] = df["x"] * df["y"] * df["z"] / (df["x"].mean() * df["y"].mean() * df["z"].mean())
    df["length_ratio"] = df["x"] / df["x"].mean()
    df["width_ratio"] = df["y"] / df["y"].mean()
    df["height_ratio"] = df["z"] / df["z"].mean()
    df["sphericity"] = 1.4641 * (6 * df["volume"])**(2/3) / df["surface_area"]
    df["compactness"] = df["volume"]**(1/3) / df["x"]
    
    return df


# In[103]:


cut_dic = {'Fair':0,'Good':1,'Very Good':2,'Premium':3,'Ideal':4}
color_dic = {'D':6,'E':5,'F':4,'G':3,'H':2,'I':1,'J':0}
clarity_dic = {'FL':10, 'IF':9, 'VVS1':8, 'VVS2':7, 'VS1':6, 'VS2':5, 'SI1':4, 'SI2':3, 'I1':2, 'I2':1, 'I3':0}
train_df = process(train)
test_df = process(test)


# In[104]:


train.shape , test.shape


# In[107]:


y = train.price


# In[108]:


x_train = train.drop(columns = "price", axis = 1)


# In[109]:


x_train.shape


# In[113]:


x_train, x_val, y_train, y_val = train_test_split(x_train, y, test_size=0.2, random_state=0)


# In[122]:



lgbm_params = {
        'task': 'train',
        'objective': "regression",
        'metric': "rmse",
        'boosting_type': 'gbdt',
        'learning_rate': 0.0005,
        'num_iterations': 300000,
        'max_depth': -1,
        'feature_pre_filter': False,
        'lambda_l1': 2.877895439833595e-06,
        'lambda_l2': 0.00046039862026592493,
        'num_leaves': 31,
        'feature_fraction': 0.7,
        'bagging_fraction': 1.0,
        'bagging_freq': 0,
        'min_child_samples': 50,
        'verbosity': -1
    }
lgb_train = lgbm.Dataset(x_train, y_train)
lgb_eval = lgbm.Dataset(x_val, y_val, reference=lgb_train)
evaluation_results = {}                       


# In[124]:


evaluation_results = {}                       
model = lgbm.train(
    lgbm_params,
    valid_names=['train', 'valid'],           
    valid_sets=[lgb_train, lgb_eval],     
    evals_result=evaluation_results,      
    train_set=lgb_train,
    early_stopping_rounds=1000,
    verbose_eval=1000
)


# In[125]:


catboost_clf = catboost.CatBoostRegressor(n_estimators=10000,random_state=100,verbose= 100, loss_function='RMSE',eval_metric="RMSE")

catboost_clf.fit(x_train, y_train,
                 early_stopping_rounds=100, 
             eval_set=[(x_val, y_val)])


# In[127]:


lgbm_preds = model.predict(test)
cat_preds = catboost_clf.predict(test)
final_preds = np.column_stack([lgbm_preds, cat_preds]).mean(axis=1)


# In[132]:


splgm["price"] = lgbm_preds
splgm.to_csv("lgbm.csv" , index = False)


# In[133]:


spcat["price"] = cat_preds
spcat.to_csv("cat.csv")


# In[134]:


sp["price"] = final_preds
sp.to_csv("ensemble.csv" , index = False)


# In[142]:


submission2 = pd.read_csv("cat.csv").drop(columns = "Unnamed: 0" )


# In[137]:


pd.read_csv("ensemble.csv")


# In[140]:


sp.to_csv("submission.csv",index= 0)


# In[141]:


pd.read_csv("submission.csv")


# In[148]:


submission2.to_csv("submission2.csv",index= 0)


# In[ ]:




