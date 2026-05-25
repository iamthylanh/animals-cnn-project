from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

app = Flask(__name__)

# LOAD MODEL
model = tf.keras.models.load_model("best_model.keras")

# LOAD CLASS NAMES
with open("class_names.txt", "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines()]

IMG_SIZE = (224, 224)

# HOME PAGE
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
        return "Chưa chọn ảnh!"

    try:
        # Đọc ảnh
        image = Image.open(file).convert("RGB")

        # Resize
        image = image.resize(IMG_SIZE)

        # Convert sang numpy
        image_array = np.array(image) / 255.0

        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)

        # Predict
        prediction = model.predict(image_array)

        predicted_index = np.argmax(prediction)

        predicted_class = class_names[predicted_index]

        confidence = float(np.max(prediction)) * 100

        return render_template(
            "index.html",
            prediction=predicted_class,
            confidence=f"{confidence:.2f}"
        )

    except Exception as e:
        return f"Lỗi: {e}"

# RUN APP
if __name__ == "__main__":
    app.run(debug=True)