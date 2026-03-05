import pandas as pd
import numpy as np
import os

# 1. Load data
df = pd.read_csv(r"C:\Users\Pranav\OneDrive\Desktop\Project\Data\Used Car Dataset.csv")

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Missing values:")
print(df.isnull().sum())

# 2. Clean data
# Remove rows where price is missing
df = df.dropna(subset=["price(in lakhs)"])

# Remove duplicates
df = df.drop_duplicates()

# Remove unrealistic price values
df = df[df["price(in lakhs)"] > 0]                    # price > 0
df = df[df["price(in lakhs)"] < 200]                  # price < 200 lakhs

# Remove unrealistic kms_driven
df = df[df["kms_driven"] >= 0]                        # kms >= 0
df = df[df["kms_driven"] <= 500000]                   # kms <= 500000

# 3. Create new features
CURRENT_YEAR = 2026  # change if needed
df["car_age"] = CURRENT_YEAR - df["registration_year"]

# Optional: price per km
df["price_per_km"] = df["price(in lakhs)"] * 100000 / df["kms_driven"]
df = df[df["price_per_km"] < 100000]  # cap unrealistic price_per_km

print("\nAfter cleaning:")
print("Shape:", df.shape)
print("New columns:", df[["car_age", "price_per_km"]].describe())

# 4. Save cleaned data
os.makedirs("../data", exist_ok=True)
df.to_csv(r"C:\Users\Pranav\OneDrive\Desktop\Project\data\used_cars_cleaned.csv", index=False)
print("Cleaned data saved to ../data/used_cars_cleaned.csv")

# 5. Visualizations (Matplotlib + Seaborn)
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# Create plots folder
os.makedirs("../plots", exist_ok=True)

# 5.1 Price distribution
plt.figure(figsize=(10, 5))
sns.histplot(df["price(in lakhs)"], bins=30, kde=True)
plt.title("Distribution of Used Car Prices")
plt.xlabel("Price (in lakhs)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(r"C:\Users\Pranav\OneDrive\Desktop\Project\price_distribution.png", dpi=200, bbox_inches="tight")
plt.show()

# 5.2 Price by fuel type
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x="fuel_type", y="price(in lakhs)")
plt.title("Car Price by Fuel Type")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(r"C:\Users\Pranav\OneDrive\Desktop\Project\price_by_fuel.png", dpi=200, bbox_inches="tight")
plt.show()

# 5.3 Price vs kms_driven
plt.figure(figsize=(10, 5))
sns.scatterplot(data=df, x="kms_driven", y="price(in lakhs)", alpha=0.7)
plt.title("Price vs KMs Driven")
plt.xlabel("KMs Driven")
plt.ylabel("Price (in lakhs)")
plt.tight_layout()
plt.savefig(r"C:\Users\Pranav\OneDrive\Desktop\Project\price_vs_kms.png", dpi=200, bbox_inches="tight")
plt.show()

# 5.4 Price vs car_age
plt.figure(figsize=(10, 5))
sns.scatterplot(data=df, x="car_age", y="price(in lakhs)", alpha=0.7)
plt.title("Price vs Car Age (Years)")
plt.xlabel("Car Age (Years)")
plt.ylabel("Price (in lakhs)")
plt.tight_layout()
plt.savefig(r"C:\Users\Pranav\OneDrive\Desktop\Project\price_vs_age.png", dpi=200, bbox_inches="tight")
plt.show()

# 5.5 Top 10 brands by average price
top_brands = (
    df.groupby("car_name")["price(in lakhs)"]
    .agg(["mean", "count"])
    .reset_index()
    .sort_values("mean", ascending=False)
    .head(10)
)

plt.figure(figsize=(10, 6))
sns.barplot(data=top_brands, x="mean", y="car_name")
plt.title("Top 10 Cars by Average Price")
plt.xlabel("Average Price (in lakhs)")
plt.ylabel("Car Brand/Model")
plt.tight_layout()
plt.savefig(r"C:\Users\Pranav\OneDrive\Desktop\Project\Top 10 Cars_Average_Price.png", dpi=200, bbox_inches="tight")
plt.show()



