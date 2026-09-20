
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load historical sales data
data = pd.read_csv("sales_data.csv")
data["Month"] = pd.to_datetime(data["Month"])

# Basic analysis
average_sales = data["Sales"].mean()
highest_sales = data.loc[data["Sales"].idxmax()]
lowest_sales = data.loc[data["Sales"].idxmin()]

print("Average Monthly Sales:", round(average_sales, 2))
print("Highest Sales Month:", highest_sales["Month"].strftime("%Y-%m"))
print("Highest Sales:", highest_sales["Sales"])
print("Lowest Sales Month:", lowest_sales["Month"].strftime("%Y-%m"))
print("Lowest Sales:", lowest_sales["Sales"])

# Monthly sales trend
plt.figure(figsize=(9, 5))
plt.plot(data["Month"], data["Sales"], marker="o")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Monthly Sales Trend")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("sales_trend.png")
plt.close()

# Linear Regression forecasting
x = np.arange(len(data))
y = data["Sales"].values

slope, intercept = np.polyfit(x, y, 1)

# Forecast next 6 months
future_x = np.arange(len(data), len(data) + 6)

future_dates = pd.date_range(
    data["Month"].iloc[-1] + pd.offsets.MonthBegin(1),
    periods=6,
    freq="MS"
)

forecast = slope * future_x + intercept

# Create forecast DataFrame
forecast_df = pd.DataFrame({
    "Month": future_dates.strftime("%Y-%m"),
    "Forecast_Sales": np.round(forecast, 2)
})

forecast_df.to_csv("sales_forecast.csv", index=False)

# Forecast chart
plt.figure(figsize=(9, 5))

plt.plot(
    data["Month"],
    data["Sales"],
    marker="o",
    label="Historical Sales"
)

plt.plot(
    future_dates,
    forecast,
    marker="o",
    linestyle="--",
    label="Forecast"
)

plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Sales Forecast Using Linear Regression")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("sales_forecast.png")
plt.close()

# Display forecast
print("\nForecast for next 6 months:")
print(forecast_df)

print("\nForecasting completed successfully.")
