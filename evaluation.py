import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from preprocessing import load_data, preprocess_dataset

# LOAD MODEL
model = tf.keras.models.load_model("best_model.keras")

# LOAD DATASET
raw_train, raw_val, raw_test = load_data()
train_ds, val_ds, test_ds = preprocess_dataset(
    raw_train,
    raw_val,
    raw_test
)

# CLASS NAMES
class_names = raw_train.class_names

# EVALUATE MODEL
print("\n=== MODEL EVALUATION ===")

loss, accuracy = model.evaluate(test_ds)

print(f"\nTest Loss     : {loss:.4f}")
print(f"Test Accuracy : {accuracy:.4f}")

# PREDICTION
y_true = []
y_pred = []

for images, labels in test_ds:
    predictions = model.predict(images, verbose=0)

    predicted_labels = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_labels)

# CLASSIFICATION REPORT
print("\n=== CLASSIFICATION REPORT ===\n")

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names
)

print(report)

# CONFUSION MATRIX
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()
plt.show()