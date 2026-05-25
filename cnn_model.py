import tensorflow as tf
from tensorflow.keras import layers, models

def create_cnn_model(input_shape=(128, 128, 3), num_classes=10):
    """
    Hàm khởi tạo cấu trúc mạng CNN.
    Sử dụng các lớp: Conv2D, BatchNormalization, MaxPooling2D, và Dropout.
    """
    model = models.Sequential(name="Custom_CNN_Model")
    
    # -------------------------------------------------------------
    # Trích xuất đặc trưng cơ bản
    # -------------------------------------------------------------
    model.add(layers.Input(shape=input_shape))
    
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())  
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))         
    
    # -------------------------------------------------------------
    # Trích xuất đặc trưng trung cấp
    # -------------------------------------------------------------
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))
    
    # -------------------------------------------------------------
    # Trích xuất đặc trưng nâng cao (Sâu hơn)
    # -------------------------------------------------------------
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.3))
    
    # -------------------------------------------------------------
    # KẾT NỐI (FULLY CONNECTED LAYERS) & PHÂN LOẠI
    # -------------------------------------------------------------
    model.add(layers.Flatten())             
    
    # Tầng ẩn phân tích đặc trưng sâu
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    
    # Tầng đầu ra (Phân loại 10 lớp con vật)
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model

# ─────────────────────────────────────────────
# 3. Tóm tắt model & trực quan hoá
# ─────────────────────────────────────────────
def print_model_summary(model: tf.keras.Model) -> None:
    model.summary(line_length=75)
    total_params = model.count_params()
    trainable    = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"\nTotal params     : {total_params:,}")
    print(f"Trainable params : {trainable:,}")


def visualize_model(model: tf.keras.Model, save_path: str = "model_architecture.png") -> None:
    """Lưu sơ đồ kiến trúc model ra file PNG."""
    try:
        tf.keras.utils.plot_model(
            model,
            to_file=save_path,
            show_shapes=True,
            show_layer_names=True,
            dpi=96,
            expand_nested=True,
        )
        print(f"Model diagram saved → {save_path}")
    except Exception as e:
        print(f"Cannot save diagram (pydot/graphviz needed): {e}")
