import json
import os

# Path to images (folders where each folder contains butterflies of a different type)
train_images_path = r"D:\train_sorted"

# Get folder names = class names
class_names = sorted([d for d in os.listdir(train_images_path) if os.path.isdir(os.path.join(train_images_path, d))])

print(f"Number of classes: {len(class_names)}")
print("Class names:", class_names)

# Write them to a JSON file inside the model folder
with open("model/class_names.json", "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=4)

print("class_names.json has been successfully written inside the model folder!")