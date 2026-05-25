import tensorflow as tf
import matplotlib.pyplot as plt
from cnn_model import create_cnn_model
from preprocessing import load_data, preprocess_dataset
from augmentation import apply_augmentation

def main():
    # 1. Tải và tiền xử lý dữ liệu
    raw_train, raw_val, raw_test = load_data()
    train_ds, val_ds, test_ds = preprocess_dataset(raw_train, raw_val, raw_test)
    train_ds = apply_augmentation(train_ds)

    # 2. Khởi tạo mô hình
    model = create_cnn_model(input_shape=(224, 224, 3), num_classes=10)

    # 3. COMPILE MODEL
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 4. CẤU HÌNH CALLBACKS (EARLYSTOPPING & MODELCHECKPOINT)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=5, 
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='best_model.keras', 
            save_best_only=True, 
            monitor='val_loss'
        )
    ]

    # 5. TRAIN MODEL
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=50,
        callbacks=callbacks
    )

    # 6. ACCURACY/LOSS (Vẽ đồ thị)
    plot_accuracy_loss(history)

def plot_accuracy_loss(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Train Acc')
    plt.plot(val_acc, label='Val Acc')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.title('Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()