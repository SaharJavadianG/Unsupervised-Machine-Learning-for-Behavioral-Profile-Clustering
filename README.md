# Unsupervised-Machine-Learning-for-Behavioral-Profile-Clustering
# Small Research Case Study: Unsupervised Machine Learning for Behavioral Profile Clustering

## Project Overview

This project explores whether unsupervised machine learning can discover meaningful behavioral population segments without using clinical diagnosis labels during training.

Instead of relying only on single health indicators such as BMI, the analysis uses lifestyle, behavioral, and biometric features to identify naturally occurring groups within the data. The final model applies **K-Means clustering** to create interpretable profiles that can support prevention-focused health and wellness strategies.

## Objective

The goal of this project is to:

- Build an unsupervised clustering pipeline.
- Prevent data leakage by excluding the clinical target label during training.
- Remove low-information features that add noise to distance-based clustering.
- Standardize numeric features so no single scale dominates the model.
- Identify meaningful behavioral personas using cluster centroids.
- Translate unsupervised clusters into stakeholder-friendly insights.

## Dataset Context

The dataset contains lifestyle-related variables. The target column, `NObeyesdad`, is used only for ground-truth isolation and is not included in model training.

Key feature categories include:

- Body measurements
- Eating habits
- Physical activity
- Technology usage
- Lifestyle indicators

## Methodology

### 1. Select features and isolate the target (Y) since it is unsupervised

The clinical classification label is separated before modeling to avoid data leakage.

```python
y_ground_truth = df["NObeyesdad"].copy()

columns_to_drop = [
    "NObeyesdad",
    "SMOKE",
    "SCC"
]

df_features = df.drop(columns=columns_to_drop, errors="ignore")
```

`SMOKE` and `SCC` are removed because they have low information value

### 2. Variance Audit

Numeric feature variance is inspected before scaling.

```python
num_df = df_features.select_dtypes(include=["int64", "float64"])
print(num_df.var().sort_values(ascending=False))
```

This step confirms that features such as `Weight` and `Age` have much larger raw variance than bounded survey-style features, making standardization necessary.

### 3. Categorical Encoding

Categorical variables are routed into two encoding tracks.

Balanced categorical variables are one-hot encoded:

```python
df_features = pd.get_dummies(df_features, columns=balanced_cols, dtype=int)
```

Highly imbalanced categorical variables are frequency encoded:

```python
from category_encoders import CountEncoder

ce_freq = CountEncoder(cols=imbalanced_cols, normalize=True)
df_features = ce_freq.fit_transform(df_features)
```

This avoids creating sparse high-dimensional columns for skewed categories.

### 4. Correlation Audit

A Pearson correlation heatmap is used to inspect relationships among features.

```python
plt.figure(figsize=(18, 16))
sns.heatmap(df_features.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Interaction Matrix Excluding Target")
plt.show()
```

This helps identify possible multicollinearity while confirming that the target label is excluded from the training matrix.

### 5. Feature Standardization

Continuous numeric features are standardized to mean 0 and variance 1.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = df_features.copy()
df_scaled[num_cols] = scaler.fit_transform(df_scaled[num_cols])
```

This is essential because K-Means uses Euclidean distance, which is sensitive to feature scale.

### 6. K Selection with the Elbow Method

The model tests multiple values of `K` and tracks inertia.

The elbow curve indicates that `K = 3` provides a strong balance between cluster separation and model simplicity.

### 7. K-Means Clustering

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
df_scaled["cluster"] = kmeans.fit_predict(df_scaled)

cluster_means = df_scaled.groupby("cluster").mean().reset_index()
```

Cluster centroids are extracted to interpret the average profile of each group.

### 8. High-Discrimination Feature Visualization

The most important separating features are selected by measuring variance across cluster centers.

```python
from pandas.plotting import parallel_coordinates

importance = cluster_means.drop("cluster", axis=1).var().sort_values(ascending=False)
top_features = importance.head(8).index

cluster_means_small = cluster_means[["cluster"] + list(top_features)]

plt.figure(figsize=(18, 6))
parallel_coordinates(cluster_means_small, "cluster")
plt.show()
```

This produces a cleaner persona-level visualization by focusing only on the strongest cluster differentiators.

## Emerging Behavioral Personas

### Cluster 0: High-Risk / Sedentary Profile

This group shows above-average weight and family history indicators, along with lower meal regularity and lower physical activity. The pattern suggests irregular eating behavior and elevated lifestyle risk.

### Cluster 1: Balanced Control Baseline

This group stays close to the global average across most features. It represents a stable baseline population with more balanced lifestyle patterns.

### Cluster 2: Younger Tech-Centric Profile

This group is younger and lower in weight on average but shows higher sedentary indicators, especially related to technology usage.

## Strategic Value

This analysis can help stakeholders in three ways:

1. **Proactive risk stratification**  
   Behavioral clusters can help identify higher-risk groups before clinical outcomes become severe.

2. **More targeted outreach**  
   The high-risk group appears connected to irregular meal structure, suggesting that interventions should focus on meal consistency rather than only calorie reduction.

3. **Data-driven target personas**  
   The balanced cluster can serve as a practical reference profile for wellness programs and behavioral nudges.

## Technologies Used

- Python
- pandas
- matplotlib
- seaborn
- scikit-learn
- category_encoders
- K-Means clustering
- StandardScaler
- Parallel coordinates visualization

## Project Structure

```text
.
├── README.md
├── notebook.ipynb
├── data/
│   └── dataset.csv
└── outputs/
    └── visualizations/
```

## How to Run

1. Clone the repository.

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

2. Install dependencies.

```bash
pip install pandas matplotlib seaborn scikit-learn category-encoders
```

3. Open the notebook or run the Python script containing the clustering pipeline.

```bash
jupyter notebook
```

4. Execute the workflow in order:

```text
Load data
Clean and filter features
Encode categorical variables
Standardize numeric variables
Run elbow method
Train K-Means model
Interpret clusters
Visualize personas
```

## Key Takeaways

- K-Means clustering can uncover meaningful behavioral segments without using clinical labels during training.
- Standardization is required because Euclidean distance is highly sensitive to scale.
- Removing low-entropy variables improves the quality of the clustering space.
- Cluster centroids make unsupervised results easier to translate into human-readable personas.
- The final three-cluster solution offers practical insights for prevention-focused wellness strategy.

## Acknowledgment

Documentation optimization were assisted by Gemini 3.
