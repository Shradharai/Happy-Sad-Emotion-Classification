# 😄 Face Emotion Classifier (Happy or Sad)

## 🧠 Overview
This project implements a Deep Learning-based **Face Emotion Classifier** that detects whether a person’s facial expression is **Happy** or **Sad** using **Convolutional Neural Networks (CNNs)**. It leverages **TensorFlow/Keras**, **OpenCV**, and **Scikit-learn** to build, train, and evaluate multiple CNN architectures, achieving near-perfect accuracy in emotion recognition tasks.

---

## 🎯 Objective
The goal of this project is to train a **supervised deep learning model** that can classify human facial expressions into two classes:
- 😄 Happy  
- 😢 Sad  

This problem is a **binary classification task**, where the model predicts one of two possible categories.

---

## ⚙️ Technologies Used
- Python  
- TensorFlow / Keras  
- OpenCV  
- NumPy, Matplotlib  
- Scikit-learn  

---
## 📊 Dataset
- Contains 514 images divided into two folders: `happy/` and `sad/`.
- Loaded using TensorFlow’s `image_dataset_from_directory()` function.
- Data automatically labeled (0 for Happy, 1 for Sad).
- Preprocessing included:
  - **Scaling pixel values** from [0,255] → [0,1]  
  - **Splitting** dataset into 70% Training, 20% Validation, and 10% Testing.

---

## 🧠 Model Architectures
Five different CNN architectures were trained and compared to analyze performance and generalization:

### Model 1 – Basic CNN
A simple baseline model with three convolution layers and one dense layer.  
**Pros:** Fast and lightweight.  
**Cons:** Prone to overfitting due to no dropout.

### Model 2 – Deeper CNN with Dropout (0.5)
Added more filters and dropout regularization to prevent overfitting.  
**Pros:** Stronger feature extraction and stability.

### Model 3 – VGG-style Double Conv Layers
Used two convolution layers per block for richer feature learning.  
**Pros:** Better pattern recognition.  
**Cons:** Slower training due to complexity.

### Model 4 – Light CNN with High Dropout (0.7)
Simplified architecture with aggressive regularization.  
**Pros:** Excellent balance between bias and variance.  
**Result:** Best performance overall.

### Model 5 – Mixed Kernel Sizes (3x3 and 5x5)
Combined small and large filters for multi-scale feature extraction.  
**Pros:** Captures detailed and broader spatial features.  
**Cons:** Heavy and slower without significant accuracy gain.

---

## ⚙️ Model Compilation
All models were compiled using:
- **Optimizer:** Adam  
- **Loss Function:** Binary Crossentropy (for binary classification)  
- **Metric:** Accuracy  

The binary crossentropy loss measures how close the model’s predicted probabilities are to the actual binary labels (0 or 1).

---

## 🧩 Training Details
- **Epochs:** 20  
- **Batch Size:** 32  
- **Callbacks:** TensorBoard for real-time visualization  

Training showed a steady decrease in loss and increase in accuracy across all models. Model 4 achieved the best validation performance with consistent generalization.

---

## 📈 Evaluation Metrics
The model performance was evaluated using:
- **Binary Accuracy:** Measures overall correctness of predictions.
- **Precision:** Fraction of correctly predicted positive cases.
- **Recall:** Fraction of actual positives correctly identified.

**Best Model (Model 4) Results:**
| Metric | Score |
|---------|--------|
| Precision | 0.97 |
| Recall | 0.97 |
| Accuracy | 0.98 |

---

## 🧪 Testing on Custom Images
Custom test images were used to verify predictions:
```python
img = cv2.imread('test/face.jpg')
resize = tf.image.resize(img, (256,256))
yhat = model.predict(np.expand_dims(resize/255, 0))

if yhat > 0.5:
    print("Predicted: Sad 😢")
else:
    print("Predicted: Happy 😄")
