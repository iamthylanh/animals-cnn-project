import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import load_data, preprocess_dataset
from cnn_model import create_cnn_model
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# CẤU HÌNH
IMG_SIZE = (224, 224)
NUM_CLASSES = 10

plt.rcParams["font.family"] = "DejaVu Sans"

# LOAD MODEL
print("ĐANG KHỞI TẠO MÔ HÌNH CNN...")

model = create_cnn_model(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    num_classes=NUM_CLASSES
)

print("-> Khởi tạo mô hình thành công!")

print("ĐANG NẠP TRỌNG SỐ...")

try:

    model.load_weights("best_model.keras")

    model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
    )
    
    print("-> Nạp trọng số thành công!")

except Exception as e:

    print(f"Lỗi khi load weights: {e}")

    exit()

# LOAD DATASET
print("ĐANG TẢI DATASET...")

raw_train, raw_val, raw_test = load_data()

train_ds, val_ds, test_ds = preprocess_dataset(
    raw_train,
    raw_val,
    raw_test
)

class_names = raw_train.class_names

print(f"-> Số lớp động vật: {len(class_names)}")

# EVALUATE MODEL
print("MODEL EVALUATION")

loss, accuracy = model.evaluate(test_ds)

print(f"\nTest Loss     : {loss:.4f}")

print(f"Test Accuracy : {accuracy * 100:.2f}%")

# PREDICTION
print("ĐANG DỰ ĐOÁN TẬP TEST...")

y_true = []

y_pred = []

for images, labels in test_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_labels = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(labels.numpy())

    y_pred.extend(predicted_labels)

y_true = np.array(y_true)

y_pred = np.array(y_pred)

# CLASSIFICATION REPORT
print("CLASSIFICATION REPORT")

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names
)

print(report)

# CONFUSION MATRIX
print("ĐANG TẠO CONFUSION MATRIX")

cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(figsize=(12, 9))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title(
    "Confusion Matrix",
    fontsize=18,
    fontweight='bold'
)

plt.xlabel(
    "Predicted Label",
    fontsize=12
)

plt.ylabel(
    "True Label",
    fontsize=12
)

plt.xticks(rotation=45)

plt.yticks(rotation=0)

plt.tight_layout()

# SAVE FIGURE
plt.savefig(
    "confusion_matrix.png",
    dpi=300,
    bbox_inches='tight'
)

print("-> Đã lưu confusion_matrix.png")

plt.show()

# ACCURACY TỪNG LỚP
print("ĐANG TÍNH ACCURACY TỪNG LỚP...")

class_accuracy = []

for i in range(len(class_names)):

    idx = np.where(
        y_true == i
    )[0]

    acc = accuracy_score(
        y_true[idx],
        y_pred[idx]
    )

    class_accuracy.append(acc * 100)

# BIỂU ĐỒ ACCURACY
plt.figure(figsize=(12, 6))

bars = plt.bar(
    class_names,
    class_accuracy
)

plt.title(
    "Accuracy của từng lớp động vật",
    fontsize=18,
    fontweight='bold'
)

plt.xlabel("Lớp động vật")

plt.ylabel("Accuracy (%)")

plt.ylim(0, 100)

plt.xticks(rotation=30)

# HIỂN THỊ %
for bar, acc in zip(
    bars,
    class_accuracy
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{acc:.1f}%",
        ha='center'
    )

plt.tight_layout()

# SAVE FIGURE
plt.savefig(
    "class_accuracy.png",
    dpi=300,
    bbox_inches='tight'
)

print("-> Đã lưu class_accuracy.png")

plt.show()

# KẾT THÚC
print("HOÀN TẤT ĐÁNH GIÁ MÔ HÌNH")