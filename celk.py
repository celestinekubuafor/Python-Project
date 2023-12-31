import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.write('# Daily COVID-19 Cases and Deaths Over Time (Los Angeles County)')

data = pd.read_csv('LA_County_COVID_Cases_20231018.csv')

st.dataframe(data)

# Convert the 'date' column to datetime format
data['date'] = pd.to_datetime(data['date'])

# Calculate daily changes for cases and deaths in Los Angeles County
data['DailyCases_LA'] = data['new_cases'].diff()
data['DailyDeaths_LA'] = data['new_deaths'].diff()

# Calculate the average daily percentage change over the past six months
six_months_ago = pd.to_datetime('today') - pd.DateOffset(months=6)
filtered_data = data[data['date'] >= six_months_ago]

if not filtered_data.empty:
    average_daily_percentage_change_LA = filtered_data['DailyCases_LA'].pct_change().mean() * 100
    print(f"Average Daily Percentage Change in Cases (Los Angeles County): {average_daily_percentage_change_LA:.2f}%")
else:
    print("No data available for the last six months.")

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
st.pyplot(plt)
