import os
import json
import numpy as np
from PIL import Image
from mtcnn.mtcnn import MTCNN
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ----------- FACE EXTRACTION FUNCTION -------------
def extract_face_from_image(filepath, required_size=(128, 128)):
    try:
        image = Image.open(filepath).convert('RGB')
        pixels = np.asarray(image)
        detector = MTCNN()
        results = detector.detect_faces(pixels)
        if not results:
            return None
        x1, y1, width, height = results[0]['box']
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]
        face_image = Image.fromarray(face).resize(required_size)
        return np.asarray(face_image)
    except Exception as e:
        print(f"❌ Error processing image {filepath}: {e}")
        return None

# ----------- LOAD DATASET -------------------------
dataset_path = "image_classification_dataset"
class_names = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
print("✅ Class order used:", class_names)

with open("class_names.json", "w") as f:
    json.dump(class_names, f)

image_data = []
labels = []

print("📸 Extracting faces...")
for label_index, class_name in enumerate(class_names):
    folder = os.path.join(dataset_path, class_name)
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        face = extract_face_from_image(file_path)
        if face is not None:
            image_data.append(face / 255.0)
            labels.append(label_index)
        else:
            print(f"⚠️ No face detected in {file_path}")

if len(image_data) == 0:
    raise ValueError("🚫 No valid face images found.")

# ----------- SPLIT DATA --------------------------
print("🔀 Splitting data...")
X = np.array(image_data)
y = np.array(labels)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------- DATA AUGMENTATION -------------------
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

# ----------- BUILD MODEL -------------------------
print("🧠 Building model...")
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(len(class_names), activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.0005), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ----------- TRAIN MODEL -------------------------
print("🏋️ Training model...")
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=16),
    validation_data=(X_val, y_val),
    epochs=100
)

# ----------- SAVE MODEL --------------------------
model.save("person_classifier_model.h5")
print("✅ Model saved as person_classifier_model.h5")
