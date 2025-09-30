# 🌱 Plant Disease Detection using CNN (PyTorch)

This project implements a **Convolutional Neural Network (CNN)** from scratch in PyTorch to classify plant leaf images into multiple disease categories or healthy class.  
The goal is to assist in **early plant disease detection**, which is crucial for crop management and food security.  

---

## 📂 Dataset

- Source: [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) 
- The dataset contains thousands of labeled images of plant leaves.  
- Example classes include:  
  - `Apple___Black_rot`  
  - `Corn_(maize)___Northern_Leaf_Blight`  
  - `Tomato___Early_blight`  
  - `Grape___Esca_(Black_Measles)`  
  - `Pepper,_bell___healthy`  
  - and many more…  

---

## 🛠️ Project Workflow

1. **Data Preparation**
   - Dataset structured into class subfolders.
   - Data augmentation applied (resize, rotation, horizontal flip, normalization).
   - Split into **Train (80%)**, **Validation (10%)**, **Test (10%)** using `random_split`.

2. **Model Architecture**
   - CNN built **from scratch**:
     - 3 × Convolutional + ReLU + MaxPooling layers
     - Fully connected layers with dropout
     - Final softmax output for multi-class classification

3. **Training**
   - Loss: `CrossEntropyLoss`  
   - Optimizer: `Adam (lr=0.001)`  
   - Epochs: 10  
   - Monitored **training loss, validation loss, and accuracy**  

4. **Model Saving**
   - Best model checkpoint saved using validation accuracy.  
   - Final model exported as `best_model.pth`.  

5. **Evaluation**
   - Metrics: Accuracy, Precision, Recall, F1-score  
   - Confusion Matrix for class-level performance  

6. **Inference**
   - Single-image prediction with visualization.  
   - Top-3 predicted classes with probabilities.  
   - Batch prediction on a folder of test images with results saved to CSV.  

---

## 📊 Results

- Achieved **~95% validation accuracy after 10 epochs**.  
- Model generalizes well across multiple plant species and disease types.  

Example output:
[View Predictions_Output](predictions.csv)

## 🚀 How to Run

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/plant-disease-detection.git
cd plant-disease-detection

### 2. Install Requirements
pip install -r requirements.txt

### 3. Train Model
jupyter notebook Plant_Disease.ipynb

### 4. Run Web App
streamlit run app.py

Upload a leaf image and get predictions instantly in your browser.
