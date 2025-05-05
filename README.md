# Chest X-Ray Image Classification for Pneumonia Detection

## 🧠 Problem Statement

Pneumonia is a serious lung infection and can be life-threatening if not diagnosed early. Traditional manual diagnosis through X-rays is time-consuming and error-prone. This project aims to automate the process using deep learning to improve accuracy and speed of detection.

---

## 📁 Dataset

The dataset used contains labeled chest X-ray images of two categories:
- **Normal**
- **Pneumonia**

It is organized into training, testing, and validation folders. The images are grayscale, resized to a consistent shape (e.g., 150x150), and preprocessed for model input.

Dataset source: [Kaggle - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

---

## 🧰 Tools and Technologies

- Python
- TensorFlow / Keras
- NumPy, Matplotlib, Seaborn
- Jupyter Notebook

---

## 🏗️ Model Architecture

The CNN model includes:
- Multiple convolutional layers
- ReLU activation
- MaxPooling layers
- Dropout layers for regularization
- Fully connected dense layers
- Sigmoid activation in the output layer (binary classification)

Key Config:
- Input Shape: (150, 150, 1)
- Loss Function: Binary Crossentropy
- Optimizer: Adam
- Evaluation Metric: Accuracy

---

## 🚀 Training Process

- Data augmentation used to improve generalization
- Early stopping and model checkpointing applied
- Trained on GPU for faster performance
- Visualized learning curves and confusion matrix

---

## 📊 Evaluation and Results

- Achieved **accuracy > 95%** on the test set
- Low false positive and false negative rates
- Model evaluated with:
  - Accuracy
  - Precision
  - Recall
  - Confusion Matrix

---

## 🛠️ Requirements

pip install tensorflow matplotlib numpy seaborn scikit-learn

## 🔭 Future Work
- Use transfer learning (e.g., ResNet, VGG) to improve accuracy

- Deploy the model using a web or mobile interface for real-time diagnosis

- Extend to multi-class classification (e.g., bacterial vs viral pneumonia)

## 🙏 Acknowledgments
- Dataset by Paul Mooney on Kaggle

-TensorFlow/Keras community resources
