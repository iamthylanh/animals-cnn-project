import tensorflow as tf
import matplotlib.pyplot as plt

# DATA AUGMENTATION

data_augmentation = tf.keras.Sequential([

    # Flip ảnh ngang
    tf.keras.layers.RandomFlip("horizontal"),

    # Xoay ảnh
    tf.keras.layers.RandomRotation(0.1),

    # Zoom ảnh
    tf.keras.layers.RandomZoom(0.1),

], name="data_augmentation")

# HÀM ÁP DỤNG AUGMENTATION

def apply_augmentation(train_ds):

    augmented_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    return augmented_ds

# VISUALIZATION AUGMENTATION

def visualize_augmentation(dataset, class_names):

    plt.figure(figsize=(10, 10))

    for images, labels in dataset.take(1):

        augmented_images = data_augmentation(images)

        for i in range(9):

            plt.subplot(3, 3, i + 1)

            plt.imshow(augmented_images[i].numpy().astype("float32"))

            label_name = class_names[labels[i].numpy()]

            plt.title(label_name)

            plt.axis("off")

    plt.tight_layout()
    plt.show()

# TEST FILE

if __name__ == "__main__":

    from preprocessing import load_data, preprocess_dataset

    # Load dataset
    raw_train, raw_val, raw_test = load_data()

    # Lấy tên class
    class_names = raw_train.class_names

    # Preprocess
    train_ds, val_ds, test_ds = preprocess_dataset(
    raw_train,
    raw_val,
    raw_test
)

    # Apply augmentation
    train_ds = apply_augmentation(train_ds)

    print("Đã áp dụng Data Augmentation!")

    # Visualization
    visualize_augmentation(train_ds, class_names)