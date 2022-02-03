#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Step 1 unnecessary warnings during code execution
import warnings
warnings.filterwarnings('ignore')

#step 2 import pandas and numpy 
import pandas as pd
import numpy as np

# deciding decimal places
np.set_printoptions(precision=4,linewidth=100)

#matplot library for display data in graphical format
import matplotlib.pyplot as plt


# In[2]:


# step 3 import datafile and storing it into pandas dataframe(Always)
Stock_df=pd.read_csv('stock.csv')
Stock_df.head(10)


# In[3]:


# step 4 information of data
Stock_df.info()


# In[4]:


# Step 5 importing Libraries for plotting the data (first Assumtionn of LR model is linearity bet x and y)
plt.scatter(Stock_df['Nifty'],Stock_df['hdfc'])
plt.xlabel('Nifty')
plt.ylabel('hdfc')


# In[5]:


# Step 6 importing statmodel
import statsmodels.api as sm

#step 7 x is defined as independent variable (feature)
X=sm.add_constant(Stock_df['Nifty'])
X.head(5)


# In[6]:


#Step 8 Y is definded as dependent variable (Target variable)
y=Stock_df['hdfc']
y.head(5)


# In[7]:


# Step 9 importing library sklearn for ML model
from sklearn.model_selection import train_test_split


# In[8]:


# step 10 splitting dataset into train and test set
train_X,test_X,train_y,test_y=train_test_split(X,y,train_size=0.8,random_state=100)


# In[9]:


#step 11 fitting linear model equation
hdfc_lm=sm.OLS(train_y,train_X).fit()


# In[10]:


#step 12 printing result of linear regression model
print(hdfc_lm.params)


# In[11]:


#step 13 summary result of all the statistics of linear regression
hdfc_lm.summary()


# In[12]:


# step 14 importing libraries for plotting the data
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[13]:


#Step 15 importing libraries for plotting the data
get_ipython().run_line_magic('matplotlib','inline')


# In[14]:


#step 16 Check for normal distribution of error
hdfc_resid=hdfc_lm.resid
probplot=sm.ProbPlot(hdfc_resid)
plt.figure(figsize=(8,6))
probplot.ppplot(line='45')
plt.title("Normal P-P Plot of Regression Standardized Residuals")
plt.show()
#Straight line indicates cummulative Normal Distribution
#Doted line indicates cummulative Distribution of error
# DOts are closer to the straight line which indicates that res follows normids


# In[15]:


# Step 17 Test of Homoscedasticity
def get_standardized_values( vals ):
    return (vals - vals.mean())/vals.std()
plt.scatter( get_standardized_values( hdfc_lm.fittedvalues ),
    get_standardized_values( hdfc_resid ) )
plt.title( "Residual Plot: hdfc Prediction" );
plt.xlabel( "Standardized predicted values")
plt.ylabel( "Standardized Residuals");


# In[16]:


#step 18 outlier detection using Zscore
from scipy.stats import zscore


# In[17]:


Stock_df['z_score_hdfc']=zscore(Stock_df.hdfc)


# In[18]:


Stock_df[(Stock_df.z_score_hdfc > 3.0)|(Stock_df.z_score_hdfc < -3.0)]


# In[19]:


# Step 19 Outlier Detection using cook distance #
import numpy as np
hdfc_influence = hdfc_lm.get_influence() 
(c, p) = hdfc_influence.cooks_distance 
plt.stem( np.arange( len(train_X) ), 
np.round( c, 3), 
markerfmt=","); 
plt.title("Cooks distance for all observations in Finance dataset"); 
plt.xlabel("Row index") 
plt.ylabel("Cooks Distance");


# In[20]:


# Step 20 Outlier Detection using Leverage #
from statsmodels.graphics.regressionplots import influence_plot 
fig, ax = plt.subplots( figsize=(8,6)) 
influence_plot(hdfc_lm, ax = ax) 
plt.title("Leverage Value Vs Residuals") 
plt.show();


# In[21]:


#step 21 predicting on validation set
pred_y=hdfc_lm.predict(test_X)


# In[22]:


# Step 22  Strength of relatinship using R Square and error #
#Finding R-Square and RMSE#
from sklearn.metrics import r2_score, mean_squared_error


# In[23]:


np.abs(r2_score(test_y,pred_y))


# In[24]:


import numpy as np


# In[25]:


#finding root mean square error
np.sqrt(mean_squared_error(test_y,pred_y))


# In[26]:


from statsmodels.sandbox.regression.predstd import wls_prediction_std 
# Predict the y values
pred_y = hdfc_lm.predict(test_X ) 
# Predict the low and high interval values for y
_, pred_y_low, pred_y_high = wls_prediction_std( hdfc_lm,test_X,alpha = 0.1) 
# Store all the values in a dataframe
pred_y_df = pd.DataFrame({'Nifty Closing': test_X['Nifty'], 
'pred_y': pred_y, 
'pred_y_left': pred_y_low, 
'pred_y_right': pred_y_high })


# In[27]:


pred_y_df[0:10]


# In[ ]:




