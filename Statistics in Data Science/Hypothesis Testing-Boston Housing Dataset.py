#!/usr/bin/env python
# coding: utf-8

# # Intro

# The Boston Housing Dataset is a derived from information collected by the U.S. Census Service concerning housing in the area of Boston MA.

#  **Columns are as follow:**
# - CRIM - per capita crime rate by town
# - ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
# - INDUS - proportion of non-retail business acres per town.
# - CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
# - NOX - nitric oxides concentration (parts per 10 million)
# - RM - average number of rooms per dwelling
# - AGE - proportion of owner-occupied units built prior to 1940
# - DIS - weighted distances to five Boston employment centres
# - RAD - index of accessibility to radial highways
# * TAX - full-value property-tax rate per 10,000 dollar
# * PTRATIO - pupil-teacher ratio by town
# * LSTAT - % lower status of the population
# * MEDV - Median value of owner-occupied homes in 1000's dollar

# ## Project Objective:
# We need to provide information to help with making an informed decision by answering the following questions:
# 
# 1. Is there a significant difference in the median value of houses bounded by the Charles river or not?
# 2. Is there a difference in median values of houses of each proportion of owner-occupied units built before 1940?
# 3. Can we conclude that there is no relationship between Nitric oxide concentrations and the proportion of non-retail business acres per town?
# 4. What is the impact of an additional weighted distance to the five Boston employment centres on the median value of owner-occupied homes?

# * α = 0.05

# ***

# In[1]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import scipy.stats 
import statsmodels.api as sm
from statsmodels.formula.api import ols


# In[2]:


boston_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ST0151EN-SkillsNetwork/labs/boston_housing.csv'
boston_df=pd.read_csv(boston_url)


# In[3]:


boston_df.head()


# In[4]:


boston_df['RM'] = boston_df['RM'].round() #It doesn't make sense if number of room is fraction so we'll round it


# In[5]:


boston_df.head()


# In[6]:


#We need to categorize AGE into 3 groups for the test
boston_df.loc[(boston_df['AGE'] <= 35), 'age_group'] = '35 years and younger'
boston_df.loc[(boston_df['AGE'] > 35) & (boston_df['AGE'] < 70), 'age_group'] = 'between 35 and 70 years'
boston_df.loc[(boston_df['AGE'] >= 70), 'age_group'] = '70 years and older'


# In[7]:


boston_df.loc[(boston_df['CHAS']==1),'CHAS_'] = 'YES'
boston_df.loc[(boston_df['CHAS']==0),'CHAS_'] = 'NO'


# In[8]:


boston_df.isnull().sum() #CHECK FOR NAN VALES


# ## Visualization/Exploring

# In[9]:


fig = px.histogram(boston_df, x="PTRATIO", title= "Histogram of PTRATIO variable")
fig.show()


# - The graph showing us that  pupil-teacher ratio by town (PRATIO) data is left-skewed and unimodal 
# - 27% of observations have PTRATIO between 19.8 and 20.2

# In[10]:


138/len(boston_df['PTRATIO'])


# In[11]:


fig = px.histogram(boston_df, x="CHAS_", title='Counts houses bounded by Charles river')
fig.show()


# - Most of houses aren't bounded by Charles River (only 7% is bounded) 

# In[12]:


fig = px.scatter(boston_df, x="RM", y="TAX", title="Average number of rooms Vs TAX")
fig.show()


# - We can see the  considerable variation in TAX values Vs the average number of rooms
# - There's a question mark here, why there're houses have very high TAX value (>=666) regardless of number of rooms

# In[13]:


#boston_df.query("TAX >= 666")


# In[14]:


#boston_df.query("TAX >= 666")['MEDV'].hist(bins=15)


# In[15]:


fig = px.box(boston_df, y="MEDV", title ='Boxplot for Median value of owner-occupied homes')
fig.show()


# - We can see the outliers that need to be handeled (Removing), so they don't affect the tests results

# In[16]:


fig = px.box(boston_df, x="age_group", y="MEDV")
fig.show()


# - We have outliers need to be handeled in the 3 groups 

# In[17]:


fig = px.scatter(boston_df, x="NOX", y="INDUS", title="Nitric oxide concentrations Vs the proportion of non-retail business acres per town")
fig.show()


# * We can see a positive relationship 

# In[18]:


fig = ff.create_distplot([boston_df['MEDV']], ['distplot'])
fig.update_layout(title_text='MEDV variable Distribution')
fig.show()


# - MEDV is following normal distribution, which is needed for t-test

# ### Let's handle outliers 
# - Handeling will be by replacing outliers with nan values then removing them 
# 

# In[19]:


# This function does the replacing part

def replace_outliers(data, col):
    for x in [col]:
        q75,q25 = np.percentile(data.loc[:,x],[75,25])
        intr_qr = q75-q25

        max = q75+(1.5*intr_qr)
        min = q25-(1.5*intr_qr)

        data.loc[data[x] < min,x] = np.nan
        data.loc[data[x] > max,x] = np.nan


# In[20]:


replace_outliers(boston_df, "MEDV")


# In[21]:


boston_df.isnull().sum()


# In[22]:


boston_df = boston_df.dropna(axis = 0) 


# In[23]:


boston_df.isnull().sum()


# In[24]:


first_group = pd.DataFrame(boston_df[boston_df['age_group'] == '35 years and younger']['MEDV'])
second_group = pd.DataFrame( boston_df[boston_df['age_group'] == 'between 35 and 70 years']['MEDV'])
third_group = pd.DataFrame( boston_df[boston_df['age_group'] == '70 years and older']['MEDV'])


# In[25]:


#Replace outliers with nan values then remove them
replace_outliers(second_group, "MEDV")
replace_outliers(third_group, "MEDV")

second_group = second_group.dropna(axis = 0)
third_group = third_group.dropna(axis = 0)


# In[26]:


boston_df.isnull().sum()


# ## 1. Is there a significant difference in median value of houses bounded by the Charles river or not?
# ### Using T-test

# The Following assumption must be met:
# *   One independent, categorical variable with two levels or group ✓
# *   One dependent continuous variable ✓
# *   Independence of the observations. Each subject should belong to only one group. There is no relationship between the observations in each group. ✓
# *   The dependent variable must follow a normal distribution ✓
# *   Assumption of homogeneity of variance 

# Hypotheses
# - **H_0:**   µ\_1 = µ\_2 -> There is no difference between houses bounded by the Charles river and the houses which not 
# - **H_1:**   µ\_1 ≠ µ\_2 -> There is a  difference between houses bounded by the Charles river and the houses which not

# In[27]:


scipy.stats.levene(boston_df[boston_df['CHAS_'] == 'YES']['MEDV'],
                   boston_df[boston_df['CHAS_'] == 'NO']['MEDV'], center='mean')

#P-value > 0.05, and we can apply T-test now


# In[28]:


scipy.stats.ttest_ind(boston_df[boston_df['CHAS_'] == 'YES']['MEDV'],
                   boston_df[boston_df['CHAS_'] == 'NO']['MEDV'], equal_var = True)


# **Conslusion:** P-value < 0.05, then we can reject H_0, As there's significant difference between houses bounded by Charles river and houses which not

# ## 2. Is there a difference in Median values of houses (MEDV) for each proportion of owner occupied units built prior to 1940 (AGE)?
# ### Using ANOVA

# Hypotheses
# - **H_0:** µ_1 = µ_2 -> there no difference in Median values of houses (MEDV) for each proportion of owner occupied units built prior to 1940 (AGE)
# - **H_1:** µ_1 ≠ µ_2 -> there a difference in Median values of houses (MEDV) for each proportion of owner occupied units built prior to 1940 (AGE)

# Levene test for equality of variance 

# In[29]:


scipy.stats.levene(first_group['MEDV'], second_group['MEDV'], third_group['MEDV'], center='mean')
# since the p-value > 0.05, the variance are equal


# In[30]:


f_statistic, p_value = scipy.stats.f_oneway(first_group, second_group, third_group)
print("F_Statistic: {0}, P-Value: {1}".format(f_statistic,p_value))


# **Conclusion:** 
# - Since P-value is greater than 0.05 then we fail to reject the null hypothesis 
# - That means there no difference in Median values of houses (MEDV) for each proportion of owner occupied units built prior to 1940

# ## 2. Can we conclude that there is no relationship between Nitric oxide concentrations and proportion of non-retail business acres per town?
# ### Using Pearson Correlation

# Hypotheses
# - **H_0:** There's no relationship between NOX and INDUS
# - **H_1:** There's a relationship between NOX and INDUS

# Since they are both continuous variables we can use a pearson correlation test and draw a scatter plot

# In[31]:


scipy.stats.pearsonr(boston_df['NOX'], boston_df['INDUS'])


# **Conclusion:** Since the p-value  (Sig. (2-tailed) < 0.05, we reject the Null hypothesis and conclude that there a relationship between NOX and INDUS

# ## 3. What is the impact of an additional weighted distance  to the five Boston employment centres (DIS) on the median value of owner occupied homes (MEDV)?
# ### Using Regression Analysis

# In[32]:


X = boston_df['DIS']
y = boston_df['MEDV']
## add an intercept (beta_0) to our model
X = sm.add_constant(X) 

model = sm.OLS(y, X).fit()
predictions = model.predict(X)

# Print out the statistics
model.summary()


# **Conclusion:** The addational one unit of DIS cause an increase in MEDV by 1.3 

# In[33]:


fig = px.scatter(boston_df, x="DIS", y="MEDV", trendline="ols")
fig.show()

