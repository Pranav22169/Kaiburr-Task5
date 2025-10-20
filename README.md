#  Kaiburr Task 5 – Consumer Complaint Text Classification
**Author:** Pranav Biju Nair  
**Date:** October 2025  

---

##  Project Overview

This project is part of **Kaiburr Task 5**, focused on building an **end-to-end multi-class text classification system** using real-world consumer complaint data from [data.gov](https://catalog.data.gov/dataset).  

The model classifies complaints into **four distinct categories**:

| Label | Category |
|:------:|-----------|
| 0 | Credit reporting, repair, or other |
| 1 | Debt collection |
| 2 | Consumer Loan |
| 3 | Mortgage |

The pipeline demonstrates a complete machine learning workflow — from exploratory data analysis to final model deployment.

---

##  Workflow Summary

### **1️ Exploratory Data Analysis (EDA)**
- Analyzed complaint distributions across products and channels.
- Visualized most common complaint issues.
- Generated **word clouds** for each product category.
- Engineered additional text-based features like word count and average word length.
- Saved enhanced dataset: `EDA_subset_enhanced.csv`.

### **2️ Text Preprocessing & TF-IDF Feature Extraction**
- Performed thorough text cleaning:
  - Lowercasing, punctuation & digit removal
  - Stopword removal
  - Lemmatization using WordNet
- Applied **TF-IDF vectorization** (bigrams, 5000 features) to create feature vectors.
- Encoded complaint categories using LabelEncoder.
- Split dataset into training (80%) and testing (20%).
- Saved preprocessed arrays and TF-IDF vectorizer (`tfidf_vectorizer.pkl`).

### **3 Model Training**
Trained and evaluated five models using the preprocessed data:
-  Logistic Regression  
-  Naive Bayes  
-  Support Vector Machine (SVM)  
-  Random Forest Classifier  
-  XGBoost (Optimized)

Each model was trained, tested, and evaluated using standard metrics:
- Accuracy
- Precision
- Recall
- F1-Score  
Additionally, each model’s **confusion matrix** was visualized and saved.

### **4️ Model Comparison**
- Compared all models on the same test set.
- Plotted a **Model Comparison Chart** showing Accuracy, Precision, Recall, and F1-Score.
- Identified **XGBoost** as the best-performing model overall.

### **5️ Final Model Evaluation & Cross-Validation**
- Performed **5-Fold Stratified Cross-Validation** on XGBoost to confirm generalization.
- Plotted the **Final Confusion Matrix** and **Top 10 Feature Importances** using TF-IDF tokens.
- Saved visualizations in the `Visuals/` folder for documentation.

### **6️ Prediction on Unseen Complaints**
- Loaded final trained XGBoost model and TF-IDF vectorizer.
- Cleaned and transformed unseen complaint texts.
- Generated category predictions for 5 new complaints.
- Saved the results in `Outputs/final_predictions_2025-10-19_23-20-47.csv`.

---



