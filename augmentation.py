import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

def create_augmentation_pipeline():
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"), # Lật cả ngang lẫn dọc
        tf.keras.layers.RandomRotation(0.2),                  # Tăng góc xoay lên khoảng ~72 độ
        tf.keras.layers.RandomZoom(0.2),                      # Tăng biên độ thu phóng lên 20%
        tf.keras.layers.RandomContrast(0.15),                 
    ], name="data_augmentation")
    return data_augmentation

def apply_augmentation(dataset):
    """
    Hàm áp dụng tăng cường dữ liệu cho tf.data.Dataset.
    Chỉ tăng cường phần ảnh (x), giữ nguyên nhãn (y).
    """
    data_augmentation = create_augmentation_pipeline()
    
    # Áp dụng cho tập dataset thông qua map
    augmented_ds = dataset.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    return augmented_ds

def visualize_augmentation(dataset, class_names=None, save_path="augmentation_preview.png"):
    """
    Trực quan hóa một số hình ảnh trước và sau khi áp dụng Data Augmentation.
    Lấy ra 1 batch từ dataset, áp dụng augmentation nhiều lần lên cùng một ảnh để so sánh,
    sau đó vẽ và lưu biểu đồ kết quả dưới dạng ảnh PNG để báo cáo.
    """
    # Lấy 1 batch
    for images, labels in dataset.take(1):
        image = images[0]
        label = labels[0].numpy()
        
        # Lấy nhãn lớp nếu có
        class_name = class_names[label] if class_names and label < len(class_names) else f"Class {label}"
        
        data_augmentation = create_augmentation_pipeline()
        
        plt.figure(figsize=(10, 10))
        # Ảnh gốc
        plt.subplot(3, 3, 1)
        # Nếu pixel ở dạng float [0.0, 1.0], imshow hiển thị trực tiếp.
        # Nếu ở dạng float [0.0, 255.0] thì cần chia cho 255.0 hoặc chuyển sang uint8.
        img_np = image.numpy()
        if img_np.max() > 1.0:
            img_np = img_np / 255.0
        plt.imshow(img_np)
        plt.title(f"Goc: {class_name}")
        plt.axis("off")
        
        # Các phiên bản tăng cường ngẫu nhiên
        for i in range(8):
            augmented_image = data_augmentation(tf.expand_dims(image, 0), training=True)
            plt.subplot(3, 3, i + 2)
            img = augmented_image[0].numpy()
            if img.max() > 1.0:
                img = img / 255.0
            plt.imshow(img)
            plt.title(f"Tang cuong #{i+1}")
            plt.axis("off")
            
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close() 
        print(f"Đã lưu ảnh trực quan hóa tăng cường dữ liệu tại: {save_path}")

if __name__ == "__main__":
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        "dataset/train",
        image_size=(224, 224),
        batch_size=32
    )

    class_names = dataset.class_names

    visualize_augmentation(dataset, class_names)