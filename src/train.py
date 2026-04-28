import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from src.data_ingestion import load_data
from src.preprocessing import clean_text

# create folders
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# -------------------------
# load dataset
# -------------------------
df = load_data()

# update column names if needed
X = df["text"]
y = df["spam"]

# -------------------------
# split data
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------
# pipeline
# -------------------------
pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            preprocessor=clean_text,
            stop_words="english",
            max_features=5000
        )
    ),
    (
        "model",
        LogisticRegression(max_iter=1000)
    )
])

# -------------------------
# train model
# -------------------------
pipeline.fit(X_train, y_train)

# -------------------------
# predictions
# -------------------------
y_pred = pipeline.predict(X_test)

# -------------------------
# metrics
# -------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# -------------------------
# print results
# -------------------------
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))

print("\nClassification Report:\n")
print(report)

print("Confusion Matrix:\n")
print(cm)

# -------------------------
# save metrics report
# -------------------------
with open("reports/classification_report.txt", "w") as f:
    f.write(f"Accuracy : {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall   : {recall:.4f}\n")
    f.write(f"F1 Score : {f1:.4f}\n\n")

    f.write("Classification Report:\n")
    f.write(report)

    f.write("\nConfusion Matrix:\n")
    f.write(pd.DataFrame(cm).to_string())

# -------------------------
# save model
# -------------------------
joblib.dump(pipeline, "models/spam_model.pkl")

print("\nTraining Completed Successfully")