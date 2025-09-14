# 🦋 Butterfly Image Classification

## 📌 Project Overview
This project is a **Deep Learning-based Image Classification** system that classifies different species of butterflies.  
It uses **Convolutional Neural Networks (CNN)** for training and provides a simple **Flask web interface** for image upload and prediction.

---

## 🚀 Features
- Train a CNN model to classify butterfly species.
- Preprocess butterfly dataset and create class labels.
- Save trained model and class names for future predictions.
- Flask web app for uploading an image and getting real-time classification results.
- Easy to run and extend for other image classification tasks.

---

## 📂 Project Structure
├── app.py # Flask web app
├── train_model.py # Script to train the CNN model
├── create_class_names.py # Generate class labels for butterflies
├── requirements.txt # Project dependencies

yaml
Copy code

---

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Nada-prog/Butterfly-Image-Classification.git
   cd Butterfly-Image-Classification
Create a virtual environment and activate it:

bash
Copy code
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows
Install dependencies:

bash
Copy code
pip install -r requirements.txt
📊 Usage
🔹 Train the Model
Run the following command to train the butterfly classification model:

bash
Copy code
python train_model.py
🔹 Create Class Labels
bash
Copy code
python create_class_names.py
🔹 Run the Flask App
bash
Copy code
python app.py
Then open your browser and go to:

cpp
Copy code
http://127.0.0.1:5000/
Upload an image of a butterfly and get the predicted species.

📷 Screenshots
Add screenshots or GIFs of your app running here.

📦 Dependencies
Python 3.x

Flask

TensorFlow / Keras

OpenCV

NumPy

Pandas

(See requirements.txt for full list)

👩‍💻 Author
Nada Ragab
📌 AI & Deep Learning Enthusiast | Passionate about Computer Vision
