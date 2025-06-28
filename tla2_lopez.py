
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from mtcnn.mtcnn import MTCNN
import json

# ==== CONFIG ====
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
detector = MTCNN()

# ==== LOAD MODEL & CLASS NAMES ====
model = tf.keras.models.load_model('person_classifier_model.h5')

with open("class_names.json", "r") as f:
    class_names = json.load(f)

# ==== FACE EXTRACTION FUNCTION ====
def extract_face(filepath, required_size=(128, 128)):
    image = Image.open(filepath).convert('RGB')
    pixels = np.asarray(image)
    results = detector.detect_faces(pixels)
    
    if not results:
        return None  # No face detected

    # Get the largest face (helps when multiple faces exist)
    results.sort(key=lambda r: r['box'][2] * r['box'][3], reverse=True)
    x1, y1, width, height = results[0]['box']
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = x1 + width, y1 + height

    face = pixels[y1:y2, x1:x2]
    face_image = Image.fromarray(face).resize(required_size)
    return np.asarray(face_image)

# ==== ROUTES ====

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "❌ No image uploaded."
    
    file = request.files['image']
    if file.filename == '':
        return "❌ No file selected."

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Extract face
    face_array = extract_face(filepath)
    if face_array is None:
        return "❌ No face detected."

    # Prepare for prediction
    img_array = face_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100

    return render_template(
        'index.html',
        prediction=predicted_class,
        confidence=f"{confidence:.2f}%",
        image_url=filepath
    )

if __name__ == '__main__':
    app.run(debug=True)




