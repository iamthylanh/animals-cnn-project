import os
import sys
import tensorflow as tf
from pathlib import Path
import io

# Thiết lập để hỗ trợ hiển thị tiếng việt trên window ( có thể xóa)
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hỗ trợ hiển thị tiếng Việt trên Windows

#Đường dẫn đến các thư mục dữ liệu đã được chuẩn bị
train_data_dir = Path('/content/drive/MyDrive/DoAnTHDL/dataset/train')
test_data_dir = Path('/content/drive/MyDrive/DoAnTHDL/dataset/test')
val_data_dir = Path('/content/drive/MyDrive/DoAnTHDL/dataset/val') 

#Các tham số cấu hình
BATCH_SIZE = 12 # tùy chỉnh theo nhu cầu
IMG_SIZE = (128, 128)

#Hàm tải dữ liệu từ thu mục đã chuẩn bị
def load_data(img_size = IMG_SIZE, batch_size = BATCH_SIZE):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_data_dir, 
        image_size = img_size,
        batch_size = batch_size,
        label_mode = 'int',
        shuffle = True,
        seed = 123
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_data_dir,
        image_size = img_size,
        batch_size = batch_size,
        label_mode = 'int',
        shuffle = False,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_data_dir,
        image_size = img_size,
        batch_size = batch_size,
        label_mode = 'int',
        shuffle = False,
    )
    return train_ds, val_ds, test_ds

#Hàm tiền xử lý ảnh: chuẩn hóa pixel về [0, 1]
def preprocess_image(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

#Hàm áp dụng tiền xử lý cho toàn bộ dataset và tối ưu hóa hiệu suất
def preprocess_dataset(train_ds, val_ds, test_ds):
    train_ds = train_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    #Cấu hình Cache và Prefetch để tối ưu hóa hiệu suất
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return train_ds, val_ds, test_ds

# Chạy thử pipeline tiền xử lý dữ liệu
if __name__ == "__main__":
    try:
        # 1. Nạp dữ liệu thô
        raw_train, raw_val, raw_test = load_data()
        
        # Lấy tên các lớp nhãn (class names)
        classes = raw_train.class_names
        print(f"Các lớp nhãn tìm thấy: {classes}\n")
        
        # 2. Áp dụng tiền xử lý
        train_ds, val_ds, test_ds = preprocess_dataset(raw_train, raw_val, raw_test)
        
        print("=== KIỂM TRA PIPELINE DỮ LIỆU ===")
        # Lấy thử 1 batch từ tập Train để kiểm tra
        for images, labels in train_ds.take(1):
            print(f"Kích thước một batch ảnh (batch_size, height, width, channels): {images.shape}")
            print(f"Kích thước nhãn tương ứng: {labels.shape}")
            
            # Tính min và max của pixel trong batch này
            min_pixel = tf.reduce_min(images).numpy()
            max_pixel = tf.reduce_max(images).numpy()
            print(f"Giá trị pixel nhỏ nhất: {min_pixel:.4f} (Mong đợi: 0.0)")
            print(f"Giá trị pixel lớn nhất: {max_pixel:.4f} (Mong đợi: 1.0)")
            print(f"Ví dụ một số nhãn trong batch: {labels[:10].numpy()}")
            
        print("\nChúc mừng! Pipeline tiền xử lý dữ liệu của bạn hoạt động HOÀN HẢO!")
        
    except Exception as e:
        print(f"\n[LỖI] Đã xảy ra lỗi khi chạy thử: {e}")
