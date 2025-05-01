import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data provided
shot_data = {
    'Shot_Length': [3, 4, 5, 6, 7],
    'Shots_Made': [84, 88, 61, 61, 44],
    'Shots_Missed': [17, 31, 47, 64, 90],
    'Shot_Total': [101, 119, 108, 125, 134]
}

# dataframe og the data
shot_df = pd.DataFrame(shot_data)

# percentage of successful shots
shot_df['Success_Percentage'] = shot_df['Shots_Made'] / shot_df['Shot_Total'] * 100

# Print the dataframe for successes
print("Basketball Shot Data Analysis:")
print(shot_df)

# figure drawing
plt.figure(figsize=(10, 8))

# watermark 
plt.figtext(0.5, 0.95, "JOSHUA MUTIE -SCT321-C004-0647/2025", 
           ha='center', fontsize=10, color='blue', alpha=0.7, 
           bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.3))

# Create subplot layout 
plt.subplots_adjust(hspace=0.5, wspace=0.3)  # Add more space between subplots

plt.subplot(2, 2, 1)
# Bar chart with Made vs Missed by Length
bar_width = 0.3  
positions = np.arange(len(shot_df['Shot_Length']))
plt.bar(positions - bar_width/2, shot_df['Shots_Made'], bar_width, label='Made')
plt.bar(positions + bar_width/2, shot_df['Shots_Missed'], bar_width, label='Missed')
plt.xlabel('Shot Length')
plt.ylabel('Count')
plt.title('Made vs Missed by Shot Length')
plt.xticks(positions, shot_df['Shot_Length'])
plt.legend(loc='upper right', fontsize=8)  
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.subplot(2, 2, 2)
# Line chart Success Rate by Length
plt.plot(shot_df['Shot_Length'], shot_df['Success_Percentage'], marker='o', linewidth=2, markersize=8, color='green')
plt.xlabel('Shot Length')
plt.ylabel('Success %')
plt.title('Success Rate by Shot Length')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(shot_df['Shot_Length']) # x-axis
plt.ylim(0, 100)  # y-axis 0-100%

plt.subplot(2, 2, 3)
# Stacked % bar chart
made_percent = [i / j * 100 for i,j in zip(shot_df['Shots_Made'], shot_df['Shot_Total'])]
missed_percent = [i / j * 100 for i,j in zip(shot_df['Shots_Missed'], shot_df['Shot_Total'])]

plt.bar(shot_df['Shot_Length'], made_percent, label='Made %', color='green')
plt.bar(shot_df['Shot_Length'], missed_percent, bottom=made_percent, label='Missed %', color='red')
plt.xlabel('Shot Length')
plt.ylabel('Percentage')
plt.title('Made vs Missed Percentage by Length')
plt.xticks(shot_df['Shot_Length'])
plt.legend(loc='center right', fontsize=8)  
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.subplot(2, 2, 4)
# Total attempts by length
plt.bar(shot_df['Shot_Length'], shot_df['Shot_Total'], color='purple')
plt.xlabel('Shot Length')
plt.ylabel('Total Attempts')
plt.title('Total Shot Attempts by Length')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(shot_df['Shot_Length'])

plt.tight_layout(rect=[0, 0, 1, 0.93])  
plt.savefig('shot_length_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# correlation matrix visualization
plt.figure(figsize=(7, 5))

# my watermark 
plt.figtext(0.5, 0.95, "JOSHUA MUTIE -SCT321-C004-0647/2025", 
           ha='center', fontsize=10, color='blue', alpha=0.7, 
           bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.3))

# Compute correlation matrix no total column
correlation = shot_df[['Shot_Length', 'Shots_Made', 'Shots_Missed', 'Success_Percentage']].corr()
print("\nCorrelation Matrix:")
print(correlation)

# heatmap
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f', annot_kws={"size": 8})
plt.title('Correlation Matrix')
plt.tight_layout(rect=[0, 0, 1, 0.93])  
plt.savefig('shot_correlation.png', dpi=300, bbox_inches='tight')
plt.show()

# Calculate linear regression to model the relationship
from sklearn.linear_model import LinearRegression
X_length = shot_df[['Shot_Length']].values
y_made = shot_df['Shots_Made'].values
y_missed = shot_df['Shots_Missed'].values
y_success = shot_df['Success_Percentage'].values

made_model = LinearRegression().fit(X_length, y_made)
missed_model = LinearRegression().fit(X_length, y_missed)
success_model = LinearRegression().fit(X_length, y_success)

# Print regression results
print("\nLinear Regression Results:")
print(f"Shots Made = {made_model.intercept_:.2f} + ({made_model.coef_[0]:.2f} × Shot Length)")
print(f"R² for Shots Made: {made_model.score(X_length, y_made):.4f}")

print(f"\nShots Missed = {missed_model.intercept_:.2f} + ({missed_model.coef_[0]:.2f} × Shot Length)")
print(f"R² for Shots Missed: {missed_model.score(X_length, y_missed):.4f}")

print(f"\nSuccess Percentage = {success_model.intercept_:.2f} + ({success_model.coef_[0]:.2f} × Shot Length)")
print(f"R² for Success Percentage: {success_model.score(X_length, y_success):.4f}")

# Summary statistics
print("\nSummary Statistics:")
print(shot_df.describe())

# Analysis findings
print("\nBasketball Shot Analysis Findings:")
print(f"1. As Shot Length increases, thesuccess percentage decreases - with a correlation of {correlation.loc['Shot_Length', 'Success_Percentage']:.4f}")
print(f"2. The optimal Shot Length appears to be {shot_df.loc[shot_df['Success_Percentage'].idxmax(), 'Shot_Length']} with a success percentage of {shot_df['Success_Percentage'].max():.2f}%")
print(f"3. For each unit increase in Shot Length, the success percentage decreases by approximately {abs(success_model.coef_[0]):.2f}%")
print(f"4. The highest number of total shot attempts was at Length = {shot_df.loc[shot_df['Shot_Total'].idxmax(), 'Shot_Length']} ({shot_df['Shot_Total'].max()} attempts)")
