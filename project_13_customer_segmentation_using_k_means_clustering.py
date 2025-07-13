
"""Project 13. Customer Segmentation using K-Means Clustering.ipynb



Importing the Dependencies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
import seaborn as sns

"""Data Collection & Analysis"""

# loading the data from csv file to a Pandas DataFrame
customer_data = pd.read_csv("Customers_data.csv")

# Enhanced data exploration
print("Dataset Overview:")
print(f"Shape: {customer_data.shape}")
print(f"Columns: {customer_data.columns.tolist()}")
print("\nDetailed Statistics:")
print(customer_data.describe())

# Check actual column names
print("\nColumn Information:")
for i, col in enumerate(customer_data.columns):
    print(f"Index {i}: {col}")

# first 5 rows in the dataframe
customer_data.head()

# finding the number of rows and columns
customer_data.shape

# getting some informations about the dataset
customer_data.info()

# checking for missing values
customer_data.isnull().sum()

# Feature engineering - create meaningful combinations
customer_data["Spending_Income_Ratio"] = (
    customer_data.iloc[:, 4] / customer_data.iloc[:, 3]
)

# Encode categorical variables (Gender)
le = LabelEncoder()
customer_data["Gender_Encoded"] = le.fit_transform(customer_data.iloc[:, 1])

"""Choosing the Annual Income Column & Spending Score column"""

# Using enhanced features for better clustering
X_enhanced = customer_data[
    ["Gender_Encoded", "Age", "Annual Income (k$)", "Spending Score (1-100)"]
].values

# Standardize features for better clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_enhanced)

# Also keep original 2D data for comparison
X = customer_data.iloc[:, [3, 4]].values

print("Original 2D features shape:", X.shape)
print("Enhanced features shape:", X_scaled.shape)

"""Choosing the number of clusters

WCSS  ->  Within Clusters Sum of Squares
"""

# Multiple evaluation metrics for optimal clusters
wcss = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))

# Enhanced visualization with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Elbow plot
ax1.plot(k_range, wcss, marker="o")
ax1.set_title("Elbow Method for Optimal k")
ax1.set_xlabel("Number of Clusters")
ax1.set_ylabel("WCSS")
ax1.grid(True)

# Silhouette plot
ax2.plot(k_range, silhouette_scores, marker="o", color="red")
ax2.set_title("Silhouette Score vs Number of Clusters")
ax2.set_xlabel("Number of Clusters")
ax2.set_ylabel("Silhouette Score")
ax2.grid(True)

plt.tight_layout()
plt.show()

# Find optimal k
optimal_k = k_range[silhouette_scores.index(max(silhouette_scores))]
print(f"Optimal number of clusters based on Silhouette Score: {optimal_k}")

# Also show the original elbow plot for comparison
plt.figure(figsize=(8, 6))
plt.plot(
    range(1, 11),
    [
        KMeans(n_clusters=i, init="k-means++", random_state=42).fit(X).inertia_
        for i in range(1, 11)
    ],
)
plt.title("The Original Elbow Point Graph")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

"""Optimum Number of Clusters = 5

Training the k-Means Clustering Model
"""

# Train final model with optimal parameters
final_kmeans = KMeans(n_clusters=5, init="k-means++", random_state=42, n_init=10)
Y_final = final_kmeans.fit_predict(X_scaled)

# Also get clusters for original 2D data for visualization
kmeans_2d = KMeans(n_clusters=5, init="k-means++", random_state=0)
Y = kmeans_2d.fit_predict(X)

# Add cluster labels to original dataframe
customer_data["Cluster"] = Y_final

# Model evaluation metrics
final_silhouette = silhouette_score(X_scaled, Y_final)
final_inertia = final_kmeans.inertia_

print(f"Final Model Performance:")
print(f"Silhouette Score: {final_silhouette:.3f}")
print(f"Inertia (WCSS): {final_inertia:.2f}")
print(f"Number of clusters: 5")

print("Cluster labels for 2D visualization:", Y)


def analyze_customer_segments(data):
    """Comprehensive segment analysis for travel industry"""

    segment_analysis = (
        data.groupby("Cluster")
        .agg(
            {
                "Age": ["mean", "std", "min", "max"],
                "Annual Income (k$)": ["mean", "std", "min", "max"],
                "Spending Score (1-100)": ["mean", "std", "min", "max"],
                "Gender": lambda x: x.mode().iloc[0] if not x.empty else "Unknown",
                "CustomerID": "count",
            }
        )
        .round(2)
    )

    # Rename cluster size column
    segment_analysis.columns = [
        f"{col[0]}_{col[1]}" if col[1] != "" else col[0]
        for col in segment_analysis.columns
    ]
    segment_analysis = segment_analysis.rename(
        columns={"CustomerID_count": "Segment_Size"}
    )

    return segment_analysis


def create_travel_personas(data):
    """Map clusters to travel customer personas for MakeMyTrip"""

    cluster_summary = (
        data.groupby("Cluster")
        .agg(
            {
                "Age": "mean",
                "Annual Income (k$)": "mean",
                "Spending Score (1-100)": "mean",
            }
        )
        .round(0)
    )

    travel_personas = {}
    recommendations = {}

    for cluster in cluster_summary.index:
        age = cluster_summary.loc[cluster, "Age"]
        income = cluster_summary.loc[cluster, "Annual Income (k$)"]
        spending = cluster_summary.loc[cluster, "Spending Score (1-100)"]

        # Define persona based on characteristics
        if income >= 70 and spending >= 70:
            persona = "Luxury Travelers"
            rec = [
                "Premium hotels",
                "Business class flights",
                "Exclusive packages",
                "Concierge services",
            ]
        elif income <= 40 and spending <= 40:
            persona = "Budget Travelers"
            rec = [
                "Budget accommodations",
                "Economy flights",
                "Group discounts",
                "Off-season deals",
            ]
        elif age <= 35 and spending >= 60:
            persona = "Adventure Seekers"
            rec = [
                "Adventure packages",
                "Backpacking tours",
                "Extreme sports",
                "Solo travel deals",
            ]
        elif income >= 50 and spending <= 50:
            persona = "Conservative Spenders"
            rec = [
                "Family packages",
                "Safe destinations",
                "All-inclusive deals",
                "Insurance options",
            ]
        else:
            persona = "Balanced Travelers"
            rec = [
                "Mid-range hotels",
                "Flexible packages",
                "Seasonal offers",
                "Loyalty programs",
            ]

        travel_personas[cluster] = persona
        recommendations[cluster] = rec

    return travel_personas, recommendations


# Generate business insights
segment_analysis = analyze_customer_segments(customer_data)
travel_personas, recommendations = create_travel_personas(customer_data)

print("\n=== CUSTOMER SEGMENT ANALYSIS ===")
print(segment_analysis)
print("\n=== TRAVEL PERSONAS FOR MAKEMYTRIP ===")
for cluster, persona in travel_personas.items():
    size = customer_data[customer_data["Cluster"] == cluster].shape[0]
    print(f"\nCluster {cluster}: {persona} ({size} customers)")
    print("Marketing Recommendations:")
    for rec in recommendations[cluster]:
        print(f"  • {rec}")

"""5 Clusters -  0, 1, 2, 3, 4

Visualizing all the Clusters
"""

# Enhanced cluster visualization with travel personas
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

colors = ["green", "red", "yellow", "violet", "blue"]

# Original Income vs Spending plot with travel personas
for i in range(5):
    mask = Y == i
    persona_name = travel_personas.get(i, f"Cluster {i}")
    ax1.scatter(
        X[mask, 0], X[mask, 1], c=colors[i], label=persona_name, s=50, alpha=0.7
    )

# Plot centroids
ax1.scatter(
    kmeans_2d.cluster_centers_[:, 0],
    kmeans_2d.cluster_centers_[:, 1],
    c="black",
    marker="x",
    s=200,
    linewidths=3,
    label="Centroids",
)
ax1.set_title("Customer Segments: Income vs Spending Score")
ax1.set_xlabel("Annual Income (k$)")
ax1.set_ylabel("Spending Score (1-100)")
ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
ax1.grid(True, alpha=0.3)

# Age vs Income plot
for i in range(5):
    mask = customer_data["Cluster"] == i
    ax2.scatter(
        customer_data.loc[mask, "Age"],
        customer_data.loc[mask, "Annual Income (k$)"],
        c=colors[i],
        label=f"Cluster {i}",
        s=50,
        alpha=0.7,
    )
ax2.set_title("Age vs Annual Income by Cluster")
ax2.set_xlabel("Age")
ax2.set_ylabel("Annual Income (k$)")
ax2.grid(True, alpha=0.3)

# Age vs Spending plot
for i in range(5):
    mask = customer_data["Cluster"] == i
    ax3.scatter(
        customer_data.loc[mask, "Age"],
        customer_data.loc[mask, "Spending Score (1-100)"],
        c=colors[i],
        label=f"Cluster {i}",
        s=50,
        alpha=0.7,
    )
ax3.set_title("Age vs Spending Score by Cluster")
ax3.set_xlabel("Age")
ax3.set_ylabel("Spending Score (1-100)")
ax3.grid(True, alpha=0.3)

# Cluster size distribution
cluster_sizes = customer_data["Cluster"].value_counts().sort_index()
bars = ax4.bar(
    range(len(cluster_sizes)), cluster_sizes.values, color=colors[: len(cluster_sizes)]
)
ax4.set_title("Customer Distribution Across Segments")
ax4.set_xlabel("Cluster")
ax4.set_ylabel("Number of Customers")
ax4.set_xticks(range(len(cluster_sizes)))
ax4.set_xticklabels([f"C{i}" for i in cluster_sizes.index])
ax4.grid(True, alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars, cluster_sizes.values):
    ax4.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        str(value),
        ha="center",
        va="bottom",
    )

plt.tight_layout()
plt.show()

# Original simple visualization for comparison
plt.figure(figsize=(10, 8))
plt.scatter(X[Y == 0, 0], X[Y == 0, 1], s=50, c="green", label="Cluster 1")
plt.scatter(X[Y == 1, 0], X[Y == 1, 1], s=50, c="red", label="Cluster 2")
plt.scatter(X[Y == 2, 0], X[Y == 2, 1], s=50, c="yellow", label="Cluster 3")
plt.scatter(X[Y == 3, 0], X[Y == 3, 1], s=50, c="violet", label="Cluster 4")
plt.scatter(X[Y == 4, 0], X[Y == 4, 1], s=50, c="blue", label="Cluster 5")

# plot the centroids
plt.scatter(
    kmeans_2d.cluster_centers_[:, 0],
    kmeans_2d.cluster_centers_[:, 1],
    s=100,
    c="cyan",
    label="Centroids",
)

plt.title("Customer Groups - Traditional View")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Export results for business use
customer_data["Travel_Persona"] = customer_data["Cluster"].map(travel_personas)

# Create summary report
summary_report = pd.DataFrame(
    {
        "Cluster": range(5),
        "Travel_Persona": [travel_personas[i] for i in range(5)],
        "Customer_Count": [
            customer_data[customer_data["Cluster"] == i].shape[0] for i in range(5)
        ],
        "Avg_Age": [
            customer_data[customer_data["Cluster"] == i]["Age"].mean() for i in range(5)
        ],
        "Avg_Income": [
            customer_data[customer_data["Cluster"] == i]["Annual Income (k$)"].mean()
            for i in range(5)
        ],
        "Avg_Spending": [
            customer_data[customer_data["Cluster"] == i][
                "Spending Score (1-100)"
            ].mean()
            for i in range(5)
        ],
    }
).round(2)

print("\n=== BUSINESS SUMMARY REPORT ===")
print(summary_report)

# Save enhanced dataset and summary
customer_data.to_csv("customer_segments_enhanced.csv", index=False)
summary_report.to_csv("cluster_summary_report.csv", index=False)
print("\nFiles saved:")
print("- customer_segments_enhanced.csv (detailed customer data with segments)")
print("- cluster_summary_report.csv (business summary report)")

print(f"\nProject completed successfully!")
print(f"• Analyzed {customer_data.shape[0]} customers")
print(f"• Created {len(travel_personas)} distinct travel personas")
print(f"• Achieved silhouette score of {final_silhouette:.3f}")
print(f"• Generated actionable business insights for MakeMyTrip")
