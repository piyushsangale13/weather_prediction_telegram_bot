from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from io import BytesIO

# Load trained model
MODEL_PATH = "weather_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Class names (same order as used during training)
CLASS_NAMES = ["dew", "fogsmog", "frost", "glaze", "hail", "lightning", "rain", "rainbow", "rime", "sandstorm", "snow"]

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")  # Frontend page

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image file was uploaded
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    try:
        # Convert file to BytesIO and preprocess image
        img = image.load_img(BytesIO(file.read()), target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for model
        img_array = img_array / 255.0  # Normalize

        # Predict
        predictions = model.predict(img_array)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = round(100 * np.max(predictions[0]), 2)

        # Return result
        return jsonify({
            "prediction": predicted_class,
            "confidence": f"{confidence}%"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run()
