
# 🧠 Parkinson’s Disease Detection Using Ensemble Learning

## Overview

This project focuses on detecting **Parkinson’s Disease** using the **HandPD dataset**, combining **image preprocessing**, **deep feature extraction** using **DenseNet**, and powerful **ensemble classifiers**. It aims to build a robust model that achieves high accuracy while maintaining balanced performance across both healthy and Parkinson’s patients.

---

## 🗂 Dataset

- **HandPD Dataset**: A publicly available dataset consisting of hand-movement-related recordings or images labeled as either **Healthy** or **Parkinson’s**.
- **Class Imbalance**: The dataset contains an unequal number of samples across the two classes, which presents challenges in achieving balanced performance.

---

## 🛠 Preprocessing

- **Wiener Filtering**: Applied to reduce image noise and preserve edges.
- **Grayscale Conversion**: Standardizes input for consistent feature extraction across samples.

---

## 🧬 Feature Extraction

- **Model**: Pretrained **DenseNet** (e.g., DenseNet121) from `torchvision.models`.
- **Output**: High-dimensional feature vectors extracted from the final convolutional layers.
- These vectors are used as input to traditional machine learning classifiers.

---

## 🧠 Models & Ensemble Strategy

Several models were tested to find the best classifier for this task:

1. **Random Forest Classifier (RF)**  
   Tuned using GridSearchCV. Performed well but showed class imbalance in recall.

2. **Gradient Boosting Classifier (GB)**  
   Delivered the best standalone accuracy. Hyperparameters tuned using GridSearchCV.

3. **Stacking Classifier**  
   Combined RF and GB as base learners with GB as the meta-learner. Balanced but slightly lower accuracy.

4. **Voting Classifier (Final Model)**  
   Combined **Calibrated GB** and **Stacking Classifier** using **soft voting**, with **2:1 weighting** toward GB.

---

## 🔍 Final Evaluation Metrics

### ✅ Voting Classifier (Calibrated GB + Stacking, Weighted 2:1)

| Metric                         | Value       |
|-------------------------------|-------------|
| Accuracy                      | **93.28%**  |
| Balanced Accuracy             | 93.45%      |
| Macro F1-score                | 0.9327      |
| Matthews Correlation Coeff.   | 0.8676      |
| Cohen’s Kappa                 | 0.8656      |

**Classification Report** (Threshold = 0.496 for best F1):

```
              precision    recall  f1-score   support

     Healthy       0.97      0.90      0.93        63
   Parkinson       0.90      0.96      0.93        56

    accuracy                           0.93       119
   macro avg       0.93      0.93      0.93       119
weighted avg       0.93      0.93      0.93       119
```

---

## ⚠️ Challenges & Recommendations

- The dataset is **imbalanced**, leading to disparities in recall and precision across classes.
- Though stacking helped balance recall, it slightly reduced accuracy.
- The **final voting model** combines the strengths of both approaches.
- No oversampling (e.g., SMOTE) or class-weight tuning was used yet — these are **recommended** for future work.
- Threshold tuning improved F1-score, showing value in optimizing beyond default thresholds.

---

## 🧪 How to Run

1. **Preprocess** your dataset using Wiener + grayscale filters.
2. **Extract features** using a pretrained DenseNet model.
3. **Train** individual models (RF, GB) with GridSearchCV.
4. **Combine** them via stacking and soft voting.
5. **Tune threshold** for optimal F1-score using validation data.
6. **Evaluate** on the test set using accuracy, F1, MCC, and Cohen’s Kappa.

---

## 📂 Folder Structure

```
project/
│
├── feature_extraction.py       # DenseNet feature extraction
├── preprocessing.py            # Wiener + grayscale filtering
├── train_models.py             # Training RF and GB models
├── ensemble_models.py          # Stacking and voting classifiers
├── evaluate.py                 # Evaluation metrics + threshold tuning
├── requirements.txt
└── README.md                   # You are here
```

---

## 🤝 Contributions

Pull requests are welcome! If you’d like to contribute, please fork the repo and submit a PR with your improvements.

---

## 📧 Contact

For questions, reach out via GitHub issues or email: `your-email@example.com`

---

## 📝 License

This project is licensed under the MIT License — see the `LICENSE` file for details.

---

*Built with ❤️ for improving neurological disease detection through AI.*
