# STUDENT PERFORMANCE PREDICTION
# Machine Learning Project
# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  

# LOAD DATASET
df = pd.read_csv("StudentsPerformance.csv")
# Display first 5 rows
print(df.head())


# DATA CLEANING
# Check missing values
print(df.isnull().sum())

# CREATE TARGET COLUMN
# Average score
df["average_score"] = (
    df["math score"] +
    df["reading score"] +
    df["writing score"]
) / 3

# Pass = 1 , Fail = 0
df["result"] = df["average_score"].apply(
    lambda x: 1 if x >= 40 else 0
)

# ENCODE CATEGORICAL DATA
le = LabelEncoder()
df["gender"] = le.fit_transform(df["gender"])
df["race/ethnicity"] = le.fit_transform(df["race/ethnicity"])
df["parental level of education"] = le.fit_transform(
    df["parental level of education"]
)
df["lunch"] = le.fit_transform(df["lunch"])
df["test preparation course"] = le.fit_transform(
    df["test preparation course"]
)

# SELECT FEATURES AND TARGET
X = df[
    [
        "gender",
        "race/ethnicity",
        "parental level of education",
        "lunch",
        "test preparation course",
        "math score",
        "reading score",
        "writing score"
    ]
]

y = df["result"]

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# TRAIN MODEL
# Random Forest gives high accuracy
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ================================
# PREDICTION
# ================================

y_pred = model.predict(X_test)

# ================================
# ACCURACY
# ================================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy * 100)

# ================================
# CONFUSION MATRIX
# ================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:\n")
print(cm)

# Heatmap
plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()

# ================================
# CLASSIFICATION REPORT
# ================================

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))

# ================================
# FEATURE IMPORTANCE GRAPH
# ================================

importance = model.feature_importances_

features = X.columns

plt.figure(figsize=(12,8))

sns.barplot(
    x=importance,
    y=features
)

plt.title("Feature Importance")

plt.show()

# ================================
# SAMPLE PREDICTION
# ================================

sample = [[
    0,   # gender
    2,   # race/ethnicity
    3,   # parental education
    1,   # lunch
    1,   # test prep
    75,  # math
    80,  # reading
    78   # writing
]]

prediction = model.predict(sample)

if prediction[0] == 1:
    print("\nStudent Will PASS")
else:
    print("\nStudent Will FAIL")