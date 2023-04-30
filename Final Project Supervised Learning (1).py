#!/usr/bin/env python
# coding: utf-8

# # What is the Project?
# - This project is a regression analysis with the goal of prediction on the 3 major powerlifting lifts: Squats, Deadlifts, and Bench Press. 
# - To accomplish this goal, first the dataset will be cleaned, and then 3 seperate models will be created for each lift. I will use Linear Regression as a baseline, and then create Random Forest models to see if those perform better.  
# - As a powerlifter myself, though not a competitive one, it would be really neat to see what the machine learning algorithms predict I might lift. It could give me a goal for the future as well.

# ### Dataset Citation - MLA
# 
# “OpenPowerlifting Group / Openpowerlifting · GITLAB.” GitLab, https://gitlab.com/openpowerlifting/opl-data. 
# 
# Dm, H. “Powerlifting Dataset.” Kaggle, 11 Apr. 2023, https://www.kaggle.com/datasets/docgenki/powerlifting-dataset. 

# # About the Dataset
# 
# This dataset is from The OpenPowerlifting Project on Gitlab. It is a free open source dataset that includes 41 features on powerlifting, with some being demographic information, recorded lifts, and the scoring metrics on those lifts. 
# - The dataset has 17 categorical features and 24 numerical features
# - The most key information for my analysis is the best lift for deadlift, squat, and bench press. The demographic information will also be very helpful. 
# - There are over 2,000,000 records in this dataset.

# In[ ]:


# Import necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm


# In[ ]:


# Read in data
liftdata = pd.read_csv('openpowerlifting_data.csv', header = 0, low_memory = False)


# In[ ]:


print(liftdata.dtypes)


# ## Removing non-useful columns for analysis
# - Having prior knowledge of powerlifting allows me to remove some columns that would inherently cause overfitting.
# - Dots, Wilks, Glossbrenner, and Goodlift are all scores applied to lifts that have been completed. These should be removed.
# - Place fits the same idea, as the lifters are only placed after the competition portion is over.
# - Since the goal is predicting the best lift, the individual lift results can be removed for each of the 3 major lifts.
# - Name can also be removed, and TotalKg as well since this is a summation of the 3 lifts.

# In[ ]:


drop = ['Name', 'Squat1Kg','Squat2Kg', 'Squat3Kg', 'Squat4Kg', 'Bench1Kg','Bench2Kg', 'Bench3Kg', 'Bench4Kg', 'Deadlift1Kg', 'Deadlift2Kg','Deadlift3Kg', 'Deadlift4Kg', 'TotalKg', 'Place', 'Dots', 'Wilks', 'Glossbrenner', 'Goodlift']
liftdata = liftdata.drop(drop, axis = 1)


# In[ ]:


liftdata.info()


# In[ ]:


a = [np.nan, None, [], {}, 'NaN', 'Null','NULL','None','NA','?','-', '.','', ' ', '   ']

for c in liftdata.columns:
    string_null = np.array([x in a[2:] for x in liftdata[c]])
    print(c, liftdata[c].isnull().sum(), string_null.sum()) 
    
# Check Null Values


# In[ ]:


# After looking at the null data, it is very apparent that "State" has far too many null values.

# Further looking at the data, AgeClass, BirthYearClass, and WeightClassKg are redundant when we have Age and BodyweightKg.

liftdata = liftdata.drop(['State', 'AgeClass', 'BirthYearClass', 'WeightClassKg'], axis = 1)


# In[ ]:


# One feature also needs to undergo feature engineering, because it is actually a yes/no column, except no is left blank.

liftdata['Tested'] = liftdata['Tested'].fillna('No')

liftdata.Tested.replace(('Yes', 'No'), (1, 0), inplace=True)

liftdata.head()

# recheck null values

for c in liftdata.columns:
    string_null = np.array([x in a[2:] for x in liftdata[c]])
    print(c, liftdata[c].isnull().sum(), string_null.sum()) 


# In[ ]:


liftdata.corr()


# ### Correlation Matrix
# 
# - This affirms the original idea that 3 models should be made once the data is fully cleaned, since each of the lifts are closely correlated to each other. 
# - Bodyweight also appears closely correlated, but this is expected as lifters who weigh more typically lift more as well.

# In[ ]:


# At this point, we've removed unneccessary features, so now should drop rows with NA values IF that will not reduce the dataset too far.
# We will try dropNA first, and re-evaluate if needed.

liftdata1 = liftdata.dropna()


# In[ ]:


for c in liftdata1.columns:
    string_null = np.array([x in a[2:] for x in liftdata1[c]])
    print(c, liftdata1[c].isnull().sum(), string_null.sum()) 


# In[ ]:


liftdata1.info()


# # Problems?
# - This is actually incorrect for using in a final dataset, but it does tell me that dropna will be fine.
# - It is incorrect because the data first needs to be split into the 3 seperate lifts prior to dropping NA values.
# - Doing this before eliminates useful information and only focuses on lifters who competed in all 3 lifts.
# - In this dataset athletes who failed a lift are assigned a negative value. This should be adjusted to 0 since a negative lift is not possible.

# In[ ]:


squatdata = liftdata.drop(['Best3BenchKg', 'Best3DeadliftKg'], axis = 1)
benchdata = liftdata.drop(['Best3SquatKg', 'Best3DeadliftKg'], axis = 1)
deadliftdata = liftdata.drop(['Best3BenchKg', 'Best3SquatKg'], axis = 1)


# In[ ]:


squatdata = squatdata.dropna()
benchdata = benchdata.dropna()
deadliftdata = deadliftdata.dropna()


# In[ ]:


squatdata['Best3SquatKg'].mask(squatdata['Best3SquatKg'] < 0 , 0 , inplace=True)


benchdata['Best3BenchKg'].mask(benchdata['Best3BenchKg'] < 0 , 0, inplace=True)


deadliftdata['Best3DeadliftKg'].mask(deadliftdata['Best3DeadliftKg'] < 0 , 0 , inplace=True)


# In[ ]:


# Age has a problem, and actually looking into the data it looks like many ages below 10 are misclassified, so those will just be removed.

squatdata['Age'].mask(squatdata['Age'] < 10 , None , inplace=True)
squatdata = squatdata.dropna()

benchdata['Age'].mask(benchdata['Age'] < 10 , None , inplace=True)
benchdata = benchdata.dropna()

deadliftdata['Age'].mask(deadliftdata['Age'] < 10 , None , inplace=True)
deadliftdata = deadliftdata.dropna()


# In[ ]:


squatdata.info()


# In[ ]:


benchdata.info()


# In[ ]:


deadliftdata.info()


# In[ ]:


from sklearn.model_selection import train_test_split
S1 = squatdata
B1 = benchdata
D1 = deadliftdata


# In[ ]:


# Creating test/train splits

S1_train, S1_test = train_test_split(S1, test_size = 0.2, train_size = 0.8, random_state = 5)
B1_train, B1_test = train_test_split(B1, test_size = 0.2, train_size = 0.8, random_state = 5)
D1_train, D1_test = train_test_split(D1, test_size = 0.2, train_size = 0.8, random_state = 5)


# In[ ]:


# Confirm accuracy of test/train splits

print(len(S1_train) + len(S1_test) == len(S1))
print(len(B1_train) + len(B1_test) == len(B1))
print(len(D1_train) + len(D1_test) == len(D1))


# # Selecting Features
# - Initially, it should be valid to create a linear model with all features and remove the insignificant ones; however this causes a memory error, so we will create smaller models instead.
# 
# - Trying to run the models in a notebook takes over an hour, so in the interest of readability and time I will place the final model below. All other features were not statistically significant, so they would need to be removed.

# In[ ]:


S1model = smf.ols(formula='Best3SquatKg ~ Sex + Event + Equipment + Age + BodyweightKg + Tested', data=S1_train) 
resS = S1model.fit()
print(resS.summary())


# In[ ]:


B1model = smf.ols(formula='Best3BenchKg ~ Sex + Event + Equipment + Age + BodyweightKg + Tested', data=B1_train) 
resB = B1model.fit()
print(resB.summary())


# In[ ]:


D1model = smf.ols(formula='Best3DeadliftKg ~ Sex + Event + Equipment + Age + BodyweightKg + Tested', data=D1_train) 
resD = D1model.fit()
print(resD.summary())


# In[ ]:


# Check for over levereged points, or anything else to be cleaned up.

sm.graphics.plot_leverage_resid2(resS)
plt.show()


# In[ ]:


sm.graphics.plot_leverage_resid2(resB)
plt.show()


# In[ ]:


sm.graphics.plot_leverage_resid2(resD)
plt.show()


# ## Leveraged Points
# - The bench model has two levereged points, but their normalized residuals are extremely small so their impact via removal is also going to be relatively small. Therefore, we will leave them in and proceed to use the test data to verify the models.

# In[ ]:


S1Tmodel = smf.ols(formula='Best3SquatKg ~ Sex + Event + Equipment + Age + BodyweightKg + Tested', data=S1_test) 
resS2T = S1Tmodel.fit()
print(resS2T.summary())


# In[ ]:


B1Tmodel = smf.ols(formula='Best3BenchKg ~ Sex + Event + Equipment + Age + BodyweightKg + Tested', data=B1_test) 
resBT = B1Tmodel.fit()
print(resBT.summary())


# In[ ]:


D1Tmodel = smf.ols(formula='Best3DeadliftKg ~ Sex + Event + Equipment + Age + BodyweightKg + Tested', data=D1_test) 
resDT = D1Tmodel.fit()
print(resDT.summary())


# The resulting $R^2$ values are very similar between the train and test data, and this is mostly unsurprising due to the sheer volume of samples.  The test data produces a very slightly lower adjusted $R^2$ value.

# # Multicollinearity
# - It would be difficult to use these linear regression models to truly account for multicollinearity, but some of it is also part of the predictions.
# - Heavier lifters typically lift more, older lifters typically lift less etc.

# # Random Forest Model
# - This model should help the issue of multicollinearity
# - It also will hopefully provide a good basis for possible predictions.

# In[ ]:


from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# In[ ]:


# For the Random Forest Model, I will be dropping out all the data not used in the previous regression models
squatdata.info()


# In[ ]:


squatdata = squatdata.drop(['Division', 'Country', 'Federation', 'ParentFederation', 'Date', 'MeetCountry', 'MeetState', 'MeetTown', 'MeetName'], axis = 1)
benchdata = benchdata.drop(['Division', 'Country', 'Federation', 'ParentFederation', 'Date', 'MeetCountry', 'MeetState', 'MeetTown', 'MeetName'], axis = 1)
deadliftdata = deadliftdata.drop(['Division', 'Country', 'Federation', 'ParentFederation', 'Date', 'MeetCountry', 'MeetState', 'MeetTown', 'MeetName'], axis = 1)


# In[ ]:


squatdata.info()


# In[ ]:


# Need dummy vars to fit in sklearn
squatdata = pd.get_dummies(squatdata, prefix=['Sex', 'Event', 'Equip'], columns=['Sex', 'Event', 'Equipment'])
benchdata = pd.get_dummies(benchdata, prefix=['Sex', 'Event', 'Equip'], columns=['Sex', 'Event', 'Equipment'])
deadliftdata = pd.get_dummies(deadliftdata, prefix=['Sex', 'Event', 'Equip'], columns=['Sex', 'Event', 'Equipment'])


# In[ ]:


squatdata.info()


# In[ ]:


benchdata.info()


# In[ ]:


deadliftdata.info()


# In[ ]:


# Setting up feature and target variables for the 3 RF models that need to be made.
featuresS = ['Sex_F', 'Sex_M', 'Event_S', 'Event_SB', 'Event_SBD', 'Event_SD', 'Equip_Multi-ply', 'Equip_Raw', 'Equip_Single-ply', 'Equip_Straps', 'Equip_Unlimited', 'Equip_Wraps', 'Age', 'BodyweightKg', 'Tested']
featuresB = ['Sex_F', 'Sex_M', 'Event_B', 'Event_BD', 'Event_SBD', 'Event_SB', 'Equip_Multi-ply', 'Equip_Raw', 'Equip_Single-ply', 'Equip_Straps', 'Equip_Unlimited', 'Equip_Wraps', 'Age', 'BodyweightKg', 'Tested']
featuresD = ['Sex_F', 'Sex_M', 'Event_BD', 'Event_D', 'Event_SBD', 'Event_SD', 'Equip_Multi-ply', 'Equip_Raw', 'Equip_Single-ply', 'Equip_Straps', 'Equip_Unlimited', 'Equip_Wraps', 'Age', 'BodyweightKg', 'Tested']
XS = squatdata.loc [:, featuresS]
XB = benchdata.loc [:, featuresB]
XD = deadliftdata.loc [:, featuresD]
yS = squatdata.loc [:, 'Best3SquatKg']
yB = benchdata.loc [:, 'Best3BenchKg']
yD = deadliftdata.loc [:, 'Best3DeadliftKg']

# Splitting data into 3 seperate train/test sets
XS_train, XS_test, yS_train, yS_test = train_test_split(XS, yS, train_size = .8, random_state = 5)
XB_train, XB_test, yB_train, yB_test = train_test_split(XB, yB, train_size = .8, random_state = 5)
XD_train, XD_test, yD_train, yD_test = train_test_split(XD, yD, train_size = .8, random_state = 5)


# In[ ]:


regS = RandomForestRegressor(max_depth = 10, random_state = 5)
regB = RandomForestRegressor(max_depth = 10, random_state = 5)
regD = RandomForestRegressor(max_depth = 10, random_state = 5)


# In[ ]:


# Training the models... My poor CPU :(

RFS = regS.fit(XS_train, yS_train)
RFB = regB.fit(XB_train, yB_train)
RFD = regD.fit(XD_train, yD_train)


# In[ ]:


print(RFS.score(XS_test, yS_test))
print(RFB.score(XB_test, yB_test))
print(RFD.score(XD_test, yD_test))


# # Score
# - At depth 10 it fits surprisingly well in all 3 models. I will attempt to tune the hyperparameter, but bearing in mind that I will only go up to depth 15 because otherwise my computer may explode. 

# In[ ]:


max_depth_range = list(range(1, 15))
# List to store the average RMSE for each value of max_depth:
r2_list = []
for depth in max_depth_range:
    regS = RandomForestRegressor(max_depth = depth,
                            random_state = 0)
    regS.fit(XS_train, yS_train)   
    
    score = regS.score(XS_test, yS_test)
    r2_list.append(score)


# In[ ]:


fig, ax = plt.subplots(nrows = 1, ncols = 1,
                       figsize = (10,7),
                       facecolor = 'white');
ax.plot(max_depth_range,
       r2_list,
       lw=2,
       color='r')
ax.set_xlim([1, max(max_depth_range)])
ax.grid(True,
       axis = 'both',
       zorder = 0,
       linestyle = ':',
       color = 'k')
ax.tick_params(labelsize = 18)
ax.set_xlabel('max_depth', fontsize = 24)
ax.set_ylabel('R^2', fontsize = 24)
ax.set_title('Model Performance on Test Set', fontsize = 24)
fig.tight_layout()


# - I'll call it a lucky guess, but based on the graphic above a max depth of 10 does not seem unreasonable for the model. 

# In[ ]:


max_depth_range = list(range(1, 15))
# List to store the average RMSE for each value of max_depth:
r2_list = []
for depth in max_depth_range:
    regB = RandomForestRegressor(max_depth = depth,
                            random_state = 0)
    regB.fit(XB_train, yB_train)   
    
    score = regB.score(XB_test, yB_test)
    r2_list.append(score)


# In[ ]:


fig, ax = plt.subplots(nrows = 1, ncols = 1,
                       figsize = (10,7),
                       facecolor = 'white');
ax.plot(max_depth_range,
       r2_list,
       lw=2,
       color='r')
ax.set_xlim([1, max(max_depth_range)])
ax.grid(True,
       axis = 'both',
       zorder = 0,
       linestyle = ':',
       color = 'k')
ax.tick_params(labelsize = 18)
ax.set_xlabel('max_depth', fontsize = 24)
ax.set_ylabel('R^2', fontsize = 24)
ax.set_title('Model Performance on Test Set', fontsize = 24)
fig.tight_layout()


# In[ ]:


max_depth_range = list(range(1, 15))
# List to store the average RMSE for each value of max_depth:
r2_list = []
for depth in max_depth_range:
    regD = RandomForestRegressor(max_depth = depth,
                            random_state = 0)
    regD.fit(XD_train, yD_train)   
    
    score = regD.score(XD_test, yD_test)
    r2_list.append(score)


# In[ ]:


fig, ax = plt.subplots(nrows = 1, ncols = 1,
                       figsize = (10,7),
                       facecolor = 'white');
ax.plot(max_depth_range,
       r2_list,
       lw=2,
       color='r')
ax.set_xlim([1, max(max_depth_range)])
ax.grid(True,
       axis = 'both',
       zorder = 0,
       linestyle = ':',
       color = 'k')
ax.tick_params(labelsize = 18)
ax.set_xlabel('max_depth', fontsize = 24)
ax.set_ylabel('R^2', fontsize = 24)
ax.set_title('Model Performance on Test Set', fontsize = 24)
fig.tight_layout()


# Overall, for all models, a depth of 10 seems reasonable.

# In[ ]:


# Graph predicted on original data
ypredS = RFS.predict(XS_test)
ypredB = RFB.predict(XB_test)
ypredD = RFD.predict(XD_test)


# In[ ]:


x_ax = range(len(yS_test))
plt.plot(x_ax, yS_test, label="original")
plt.plot(x_ax, ypredS, label="predicted")
plt.title("Squat test and predicted data")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend(loc='best',fancybox=True, shadow=True)
plt.grid(True)
plt.show()


# In[ ]:


x_ax = range(len(yB_test))
plt.plot(x_ax, yB_test, label="original")
plt.plot(x_ax, ypredB, label="predicted")
plt.title("Bench test and predicted data")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend(loc='best',fancybox=True, shadow=True)
plt.grid(True)
plt.show()


# In[ ]:


x_ax = range(len(yD_test))
plt.plot(x_ax, yD_test, label="original")
plt.plot(x_ax, ypredD, label="predicted")
plt.title("Deadlift test and predicted data")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend(loc='best',fancybox=True, shadow=True)
plt.grid(True)
plt.show()


# # Results
# - Overall, all 3 random forest models fit significantly better than the multiple linear regression models. 
#     - This was expected due to the multicollinearity that existed previously.
#         - Visualizations are shown above and will not be redrawn to avoid clutter.
# - As shown above, multiple iterations were attempted, features dropped as needed, and some features engineered to be more useful or usable in the case of the Random Forest Models
#     - The final set of features were (Sex + Event + Equipment + Age + BodyweightKg + Tested) for a total of 6 features, which is down significantly from the 41 (38 not including test variables) features initially.
#     - I got fairly lucky in guessing that a max depth of 10 for the RF models would be good, but I was able to confirm it and so did not have to tune my hyperparameter.
# - Linear regression models
#     - $R^2$ Values:
#         - Squats: 0.595
#         - Bench: 0.650
#         - Deadlift: 0.587
# - Random Forest models
#     - $R^2$ Values:
#         - Squats: 0.683
#         - Bench: 0.745
#         - Deadlift: 0.690
# 
# - Looking at the random forest graphs of predicted vs original, all 3 predictions appear to be more centered than the actual data. This might have something to do with failed lifts being counted as 0. 
#     - A weird difference between Deadlifts and the other 2 is that the predictions appear to cap around 300. This could happen due to it trying to average in 0 values and deadlifts are typically a much higher weighted lift than the other two lifts.

# # Conclusion
# - The random forest models are the better models for use in predicting powerlifting results for any of the 3 major lifts as they fit the test data significantly better.
# - Overall, in this project I learned how to use linear regression models to identify features that could be removed due to insignificance before creating a random forest model, and also a ton of different useful tricks + plots in Python.
#     - An area of potential improvement for this model and this project is that the dataset is lacking a true classifier for the lifters. Lifters are broken into divisions, but there are far too many to be useful. It would be very helpful if lifters were classified based on their lifting experience. 
#         - Personally, I am surprised that the random forest model has an $R^2$ close to 0.7 as I would expect it to be fairly difficult to predict powerlifting results. I know personally in the last year my numbers have changed drastically, but my weight has only slightly increased. 
# - It would be really interesting to apply these models to another dataset and see if the numbers are still close.
