#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


# My file path to your CSV file on the desktop using a raw string literal
desktop_path = r'C:\Users\celes\Desktop'
file_name = 'LA_County_COVID_Cases_20231018.csv'


# In[5]:


# Combine the desktop path and file name to get the full file path
file_path = f'{desktop_path}\\{file_name}'  # You can also use os.path.join for path manipulation

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)


# In[6]:


print(df.head())


# In[7]:


print(df.columns)


# In[8]:


print(df.info())


# In[9]:


df.isnull().sum()
df = df.fillna(0) # fill NA with 0


# In[10]:


df[df.duplicated()] # check duplicates
df = df.drop_duplicates() # drop duplicates


# In[11]:


print(df.head)


# In[13]:


print(df.describe())


# In[14]:


correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Matrix')
plt.show()


# This calculates the correlation matrix for numerical variables and creates a heatmap to visualize the correlation coefficients. It helps identify relationships between different features in the dataset. (which features has impact on new death cases?)

# In[ ]:





# In[24]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Set up the Matplotlib figure size
plt.figure(figsize=(12, 6))

# Create the first histogram for 'new_cases'
sns.histplot(data=df, x='new_cases', kde=True, bins=30, color='blue', label='New Cases')

# Create the second histogram for 'new_deaths' and overlay it on the same plot
sns.histplot(data=df, x='new_deaths', kde=True, bins=30, color='red', label='New Deaths')

# Set the title of the plot
plt.title('Distribution of New Cases and New Deaths in LA')

# Set labels for the x and y axes
plt.xlabel('Counts')
plt.ylabel('Frequency')

# Display the legend to distinguish between 'new_cases' and 'new_deaths'
plt.legend()

# Display the plot
plt.show()


# The histogram visualize the distribution of new COVID-19 cases against new deaths in LA County.

# In[ ]:





# In[18]:


plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='new_state_cases', y='new_state_deaths', color='salmon')
plt.title('Scatter Plot of New State Cases vs. New State Deaths')
plt.xlabel('New State Cases')
plt.ylabel('New State Deaths')
plt.show()


# This scatter plot explore the relationship between the total state cases and the number of new deaths.

# In[ ]:





# In[28]:


sns.pairplot(df[['new_cases', 'new_state_cases','new_deaths','new_state_deaths']])
plt.show()


# A pairplot provides a matrix of scatterplots for new cases(both LAcounty & state/California cases), offering a quick overview of their relationships

# In[ ]:





# In[17]:


target_variable = 'new_deaths'
features = ['new_cases', 'state_deaths', 'state_cases']
X = df[features]
y = df[target_variable]


# In[18]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[19]:


mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')
print(f'R-squared: {r2}')


# Using mean absolute error (MAE) and R-squared metrics to evaluate the performance of the linear regression model on the test set and print the results.
# 
# The code essentially demonstrates a simple predictive analysis workflow using a linear regression model to predict the number of new deaths based on selected features from the provided COVID-19 dataset.

# In[ ]:





# In[1]:


pip install pandas matplotlib


# In[ ]:





# In[16]:


import os
import pandas as pd
import matplotlib.pyplot as plt

# Construct the full file path
desktop_path = r'C:\Users\celes\Desktop'
file_name = 'LA_County_COVID_Cases_20231018.csv'
full_file_path = os.path.join(desktop_path, file_name)

# Load your COVID-19 data into a Pandas DataFrame
data = pd.read_csv(full_file_path)

# Convert the 'Date' column to datetime format
data['date'] = pd.to_datetime(data['date'])

# Calculate daily changes for cases and deaths in Los Angeles County
data['DailyCases_LA'] = data['new_cases'].diff()
data['DailyDeaths_LA'] = data['new_deaths'].diff()

# Calculate the average daily percentage change over the past six months
six_months_ago = pd.to_datetime('today') - pd.DateOffset(months=6)
average_daily_percentage_change_LA = data[data['date'] >= six_months_ago]['DailyCases_LA'].pct_change().mean() * 100

# Print the results
print(f"Average Daily Percentage Change in Cases (Los Angeles County): {average_daily_percentage_change_LA:.2f}%")

# Visualize the data
plt.figure(figsize=(12, 8))

plt.plot(data['date'], data['DailyCases_LA'], label='Daily Cases (LA)', linestyle='-', marker='o')
plt.plot(data['date'], data['DailyDeaths_LA'], label='Daily Deaths (LA)', linestyle='-', marker='o')

# Plot rolling averages for better trend visualization
rolling_window = 7  # 7-day rolling average
plt.plot(data['date'], data['DailyCases_LA'].rolling(window=rolling_window).mean(), label=f'{rolling_window}-Day Avg Cases (LA)', linestyle='--')
plt.plot(data['date'], data['DailyDeaths_LA'].rolling(window=rolling_window).mean(), label=f'{rolling_window}-Day Avg Deaths (LA)', linestyle='--')

plt.title('Daily COVID-19 Cases and Deaths Over Time (Los Angeles County)')
plt.xlabel('Date')
plt.ylabel('Count')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45, ha='right')
plt.show()


# #The provided code generates a plot that visualizes the daily COVID-19 cases and deaths in Los Angeles County over time, along with their 7-day rolling averages. 

# In[ ]:





# In[12]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load COVID-19 data with features for severity classification
file_path = r'C:\Users\celes\Desktop\LA_County_COVID_Cases_20231018.csv'
df = pd.read_csv(file_path)

# Assuming 'new_cases' is the target variable
target_variable = 'new_cases'
features = ['new_deaths', 'new_state_deaths', 'new_state_cases']

# Extract features (X) and target variable (y)
X = df[features]
y = df[target_variable]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Display model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2%}')

# Display classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[ ]:





# <!-- ISSUES 
