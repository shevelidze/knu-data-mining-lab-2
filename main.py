import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Surprise Library for Recommendation System
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load Dataset
file_path = "ratings_Electronics.csv"
df = pd.read_csv(file_path, names=["userId", "productId", "rating", "timestamp"])

# Drop timestamp as it's not needed
df.drop("timestamp", axis=1, inplace=True)

# Filter users with at least 10 ratings
user_counts = df["userId"].value_counts()
df = df[df["userId"].isin(user_counts[user_counts >= 10].index)]

# Filter items with at least 10 ratings
item_counts = df["productId"].value_counts()
df = df[df["productId"].isin(item_counts[item_counts >= 10].index)]


# Exploratory Data Analysis (EDA)
def plot_rating_distribution():
    plt.figure(figsize=(8, 4))
    sns.histplot(df["rating"], bins=5, kde=True)
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.title("Distribution of Ratings")
    plt.show()


def calculate_sparsity():
    num_users = df["userId"].nunique()
    num_items = df["productId"].nunique()
    sparsity = 1 - (len(df) / (num_users * num_items))
    print(f"Dataset Sparsity: {sparsity:.4f}")


# User-Item Interaction Matrix Visualization
def plot_user_item_interaction():
    sample_df = df[:1000]  # Subset to avoid memory issues
    user_item_matrix = sample_df.pivot(
        index="userId", columns="productId", values="rating"
    ).fillna(0)

    plt.figure(figsize=(10, 6))
    sns.heatmap(user_item_matrix, cmap="viridis", cbar=False)
    plt.xlabel("Products")
    plt.ylabel("Users")
    plt.title("User-Item Interaction Heatmap")
    plt.show()


# Display EDA
plot_rating_distribution()
calculate_sparsity()
plot_user_item_interaction()

# Load dataset into Surprise format
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[["userId", "productId", "rating"]], reader)

# Train-test split
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Build and Train Model
model = SVD()
model.fit(trainset)

# Make Predictions
predictions = model.test(testset)

# Evaluate Model
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)
print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}")


# Function to Generate Top-N Recommendations for a Set of Users
def get_top_n(predictions, n=10, user_count=5):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    # Select a subset of users for displaying recommendations
    selected_users = list(top_n.keys())[:user_count]
    return {user: top_n[user] for user in selected_users}


# Get Recommendations for a Set of Users
top_n_recommendations = get_top_n(predictions, n=10, user_count=5)

# Display Recommendations for Multiple Users
print("\nTop 10 Recommendations for a Set of Users:")
for user, recommendations in top_n_recommendations.items():
    print(f"\nUser {user}:")
    for product, rating in recommendations:
        print(f"   Product {product} - Predicted Rating: {rating:.2f}")


# Visualization of Recommendations for Users
def plot_recommendations(top_n_recommendations):
    plt.figure(figsize=(10, 5))

    for user, recommendations in top_n_recommendations.items():
        products, ratings = zip(*recommendations)
        plt.barh(products, ratings, label=f"User {user}")

    plt.xlabel("Predicted Rating")
    plt.ylabel("Products")
    plt.title("Top-10 Recommendations for Users")
    plt.legend()
    plt.show()


plot_recommendations(top_n_recommendations)
