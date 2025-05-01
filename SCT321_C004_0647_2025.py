import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Data provided
data = {
    'Oil_Gallons': [275.30, 363.80, 164.30, 40.80, 94.30, 230.90, 366.70, 300.60, 
                    237.80, 121.40, 31.40, 203.50, 441.10, 323.00, 52.50],
    'Temperature_F': [40, 27, 40, 73, 64, 34, 9, 8, 23, 63, 65, 41, 21, 38, 58],
    'Insulation_Inches': [3, 3, 10, 6, 6, 6, 6, 10, 10, 3, 10, 6, 3, 3, 10]
}

# DataFrame
df = pd.DataFrame(data)

# confirming data
print("Data Overview:")
print(df.head())

# Basic stats
print("\nBasic Statistics:")
print(df.describe())

# Creating a multiple linear regression model
X = df[['Temperature_F', 'Insulation_Inches']]
y = df['Oil_Gallons']

model = LinearRegression()
model.fit(X, y)

# Displaying my coefficients
print("\nModel Coefficients:")
print(f"Intercept: {model.intercept_:.2f}")
print(f"Temperature Coefficient: {model.coef_[0]:.2f}")
print(f"Insulation Coefficient: {model.coef_[1]:.2f}")

# Generatinng the regression equation
equation = f"Oil Consumption = {model.intercept_:.2f} + ({model.coef_[0]:.2f} × Temperature) + ({model.coef_[1]:.2f} × Insulation)"
print(f"\nRegression Equation:\n{equation}")

# Computing model performance
y_pred = model.predict(X)
mse = np.mean((y - y_pred) ** 2)
r_squared = model.score(X, y)

print(f"\nModel Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r_squared:.2f}")

# Prediction of oil consumption for temperature = 15°F and insulation = 10 inches
new_data = np.array([[15, 10]])
predicted_consumption = model.predict(new_data)[0]
print(f"\nPredicted oil consumption for January with 15°F and 10 inches insulation: {predicted_consumption:.2f} gallons")

# Figure 1: First result 2D visualization
plt.figure(figsize=(10, 8))

# Add watermark 
plt.figtext(0.5, 0.95, "JOSHUA MUTIE -SCT321-C004-0647/2025", 
           ha='center', fontsize=12, color='blue', alpha=0.7, 
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.3))

# Createing of scatter plot with temperature on x-axis, oil on y-axis
scatter = plt.scatter(df['Temperature_F'], df['Oil_Gallons'], 
                     c=df['Insulation_Inches'], cmap='viridis', 
                     s=100, edgecolor='k')

# Adding colorbar for insulation thickness inches
cbar = plt.colorbar(scatter)
cbar.set_label('Insulation Thickness (inches)')

# Addition of regression lines for different insulation levels
insulation_levels = [3, 6, 10]
temp_range = np.linspace(0, 80, 100)

for insul in insulation_levels:
    # Generate predictions across temperature range for this insulation level
    X_pred = np.column_stack([temp_range, np.full(len(temp_range), insul)])
    y_pred = model.predict(X_pred)
    plt.plot(temp_range, y_pred, label=f'{insul} inches insulation')

# Prediction point (15°F, 10 inches)
plt.scatter([15], [predicted_consumption], color='red', s=200, 
           edgecolor='k', marker='*', label='Prediction (15°F, 10")')

# the plot
plt.xlabel('Temperature (°F)')
plt.ylabel('Oil Consumption (gallons)')
plt.title('Heating Oil Consumption vs Temperature with Different Insulation Levels')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Save the figure
plt.savefig('oil_consumption_2D.png', dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()

# Figure 2: 3D visualization of the same
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# watermark 
fig.text(0.5, 0.95, "JOSHUA MUTIE -SCT321-C004-0647/2025", 
        ha='center', fontsize=12, color='blue', alpha=0.7, 
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.3))

# Creatinf a mesh grid for temperature and insulation
temp_range = np.linspace(min(df['Temperature_F']), max(df['Temperature_F']), 20)
insul_range = np.linspace(min(df['Insulation_Inches']), max(df['Insulation_Inches']), 20)
temp_mesh, insul_mesh = np.meshgrid(temp_range, insul_range)

# Flattening the mesh grid points for prediction
points = np.vstack([temp_mesh.flatten(), insul_mesh.flatten()]).T
pred_mesh = model.predict(points).reshape(temp_mesh.shape)

# Plotting the regression surface
surf = ax.plot_surface(temp_mesh, insul_mesh, pred_mesh, alpha=0.6, cmap='viridis', edgecolor='none')

# Plot the actual data points
scatter = ax.scatter(df['Temperature_F'], df['Insulation_Inches'], df['Oil_Gallons'], 
           color='red', s=70, edgecolor='k', label='Training Data')

# Adding the predicted point (15°F, 10 inches)
ax.scatter([15], [10], [predicted_consumption], color='blue', s=200, edgecolor='k', 
          marker='*', label='Prediction (15°F, 10")')

# Custom ofthe plot
ax.set_xlabel('Temperature (°F)')
ax.set_ylabel('Insulation (inches)')
ax.set_zlabel('Oil Consumption (gallons)')
ax.set_title('3D Model of Heating Oil Consumption')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
ax.legend()

# Text annotation with the predicted value
ax.text(15, 10, predicted_consumption + 30, 
        f'Predicted: {predicted_consumption:.2f} gallons', 
        color='blue', fontsize=12)

# Save the figure for exporting to lecturer
plt.savefig('oil_consumption_3D.png', dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()

# Print the answer to part b with explanation
print("\nMy Answer to part b which is based on the regression code I made:")
print(f"A homeowner with 10 inches of attic insulation can expect to use approximately {predicted_consumption:.2f} gallons")
print("of heating oil in January when the average outside temperature is 15°F.")
print("\nThis prediction is based on the multiple linear regression model:")
print(equation)