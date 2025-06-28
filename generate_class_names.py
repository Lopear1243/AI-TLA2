import os
import json

DATA_DIR = "C:/lopez/image_classification_dataset"

class_names = sorted(os.listdir(DATA_DIR))
with open("class_names.json", "w") as f:
    json.dump(class_names, f)

print("class_names.json has been created with:", class_names)

