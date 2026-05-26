from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
import io
import base64

# Import model
from cnn_model import create_cnn_model

app = Flask(__name__)

IMG_SIZE = (224, 224)
NUM_CLASSES = 10

# LOAD MODEL
print("1. Đang khởi tạo mô hình CNN...")

model = create_cnn_model(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    num_classes=NUM_CLASSES
)

print("2. Đang nạp trọng số mô hình...")

try:
    model.load_weights("best_model.keras")
    print("-> Nạp trọng số thành công!")
except Exception as e:
    try:
        model = tf.keras.models.load_model("best_model.keras")
        print("-> Load trực tiếp mô hình thành công!")
    except Exception as e_fallback:
        print(f"Lỗi tải mô hình: {e_fallback}")

# LOAD CLASS NAMES
class_names_path = Path("class_names.txt")

if class_names_path.exists():
    with open(class_names_path, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f.readlines()]
    print(f"-> Đã nạp {len(class_names)} lớp động vật.")
else:
    class_names = [
        "bướm",
        "mèo",
        "gà",
        "bò",
        "chó",
        "voi",
        "ngựa",
        "cừu",
        "nhện",
        "sóc"
    ]
    print("-> Không tìm thấy class_names.txt")

# HOME
@app.route("/")
def home():
    return render_template("index.html")

# PREDICT
@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return "Không tìm thấy file!"

    file = request.files["file"]

    if file.filename == "":
        return "Bạn chưa chọn ảnh!"

    try:
        # ĐỌC ẢNH GỐC
        image_bytes = file.read()

        # Convert base64 để hiển thị lại ảnh
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # XỬ LÝ ẢNH CHO MODEL
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        image = image.resize(IMG_SIZE)

        image_array = np.array(image) / 255.0

        image_array = np.expand_dims(image_array, axis=0)

        # DỰ ĐOÁN
        prediction = model.predict(image_array, verbose=0)

        predicted_index = np.argmax(prediction)

        predicted_class = class_names[predicted_index]

        confidence = float(np.max(prediction)) * 100

        # RETURN
        return render_template(
            "index.html",
            prediction=predicted_class,
            confidence=f"{confidence:.2f}",
            image_data=image_base64
        )

    except Exception as e:
        return f"Lỗi xử lý ảnh: {e}"

# RUN APP
if __name__ == "__main__":
    app.run(debug=True)