#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns

#Obtaining dataset
# Load the Titanic dataset
titanic_data = sns.load_dataset('titanic')


# In[2]:


#Data exploration
# Display the first few rows of the dataset
titanic_data.head()

# Check the basic statistics of the dataset
titanic_data.describe()

# Check the data types and missing values
titanic_data.info()


# In[3]:



#Data cleaning
# Check for missing values
titanic_data.isnull().sum()

# Deal with missing values (e.g., fill or drop)
titanic_data['age'].fillna(titanic_data['age'].median(), inplace=True)
titanic_data.dropna(subset=['embarked'], inplace=True)


# In[4]:


#Data analysis
# Calculate survival rate by gender
titanic_data.groupby('sex')['survived'].mean()

# Calculate survival rate by passenger class
titanic_data.groupby('pclass')['survived'].mean()

# Calculate the average age of passengers who survived and who didn't
titanic_data.groupby('survived')['age'].mean()


# In[5]:


#Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Plot a bar chart of the survival rate by gender
sns.barplot(x='sex', y='survived', data=titanic_data)
plt.title('Survival Rate by Gender')
plt.show()

# Plot a bar chart of the survival rate by passenger class
sns.barplot(x='pclass', y='survived', data=titanic_data)
plt.title('Survival Rate by Passenger Class')
plt.show()

# Create a histogram of passenger ages
sns.histplot(titanic_data['age'], bins=20, kde=True)
plt.title('Passenger Age Distribution')
plt.show()


# In[ ]:




