import os
from flask import Flask, request, render_template, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

app = Flask(__name__)

# Load the model and compile it
model = load_model('trashnet_inceptionv3 (1).h5')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # Compile model to remove warnings
print("Expected input shape:", model.input_shape)

# Define a dictionary to map indices to disease names
disease_classes = {
    0: 'cardboard',
    1: 'glass',
    2: 'metal',
    3: 'paper',
    4: 'plastic',
    5: 'trash',
}

# Preprocess the image according to model input requirements
def preprocess_image(image):
    img = image.resize((227, 227))  # Resize to match model input size
    img = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']

    if file:
        try:
            image = Image.open(io.BytesIO(file.read()))
            processed_image = preprocess_image(image)

            # Make prediction
            prediction = model.predict(processed_image)
            predicted_class = np.argmax(prediction, axis=1)[0]
            disease_name = disease_classes.get(predicted_class, "Unknown")

            return jsonify({
                "prediction_index": int(predicted_class),
                "disease_name": disease_name
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Invalid file"}), 400

if __name__ == '__main__':
    app.run(debug=True)
