# ML & DL Projects Portfolio

A collection of end-to-end Machine Learning and Deep Learning projects covering binary/multiclass classification, multi-label NLP, regression, computer vision, and handling imbalanced data. Each notebook is self-contained, with step-by-step comments explaining the workflow — from data loading and preprocessing to model building, tuning, and evaluation.

## 📁 Projects

| # | Project | Type | Domain | Key Techniques |
|---|---------|------|--------|-----------------|
| 1 | [Distracted Driver Multi-Action Classification](DL_P1_Distracted_driver_multiaction_classification.ipynb) | Deep Learning · Multiclass Image Classification | Computer Vision / Driver Safety | CNN (Conv2D, BatchNorm, Dropout), image preprocessing with OpenCV, data augmentation-ready pipeline, confusion matrix & classification report |
| 2 | [Multiclass–Multilabel Tag Prediction for Stack Overflow Questions](DL_P2_Multiclass_Multilabel_prediction_For_stack_overflow_Questions.ipynb) | Deep Learning · Multi-label NLP | Text Classification | Tokenization & padding, dual-input GRU network (title + body embeddings), multi-label binarization, F1-score evaluation |
| 3 | [Consumer Complaints Resolution Prediction](ML_P1_Consumer_Complaints_Resolution.ipynb) | Machine Learning · Binary Classification | Finance / Customer Complaints | Feature engineering (date deltas), one-hot encoding, Logistic Regression, XGBoost, Decision Tree, SMOTE for class imbalance, ROC-AUC comparison |
| 4 | [Counterfeit Medicines Sales Prediction](ML_P3_Counterfeit_Medicines_Sales_Prediction.ipynb) | Machine Learning · Regression | Pharma / Public Safety | Missing-value imputation, one-hot encoding, hyperparameter tuning via `RandomizedSearchCV`, Decision Tree & Random Forest regressors, feature importance analysis |

## 🧠 Project Summaries

### 1. Distracted Driver Multi-Action Classification
Built a Convolutional Neural Network to classify driver behavior from dashboard camera images into 10 categories (e.g., safe driving, texting, talking on phone, drinking, reaching behind). Images were resized and normalized, the model was trained with checkpointing/early-stopping/LR-reduction callbacks, and performance was evaluated with a confusion matrix and per-class classification report.

### 2. Multiclass–Multilabel Tag Prediction for Stack Overflow Questions
Predicted relevant technology tags (e.g., Python, Java, JavaScript) for Stack Overflow questions using a dual-input GRU-based recurrent network that processes question **titles** and **bodies** separately before combining them for multi-label prediction. Evaluated using F1-score and a per-tag classification report.

### 3. Consumer Complaints Resolution Prediction
Predicted whether a consumer would dispute a company's resolution to their complaint — a binary classification problem on an imbalanced real-world dataset (~480K records). Compared Logistic Regression, XGBoost, and Decision Tree models before and after applying **SMOTE** oversampling, using ROC-AUC as the primary comparison metric.

### 4. Counterfeit Medicines Sales Prediction
Predicted counterfeit medicine sales volume using structured pharma/retail data. Performed missing-value imputation and categorical encoding, then tuned Decision Tree and Random Forest regressors via `RandomizedSearchCV`, evaluating with mean absolute error and feature importance analysis.

## 🛠️ Tech Stack

- **Languages:** Python
- **Deep Learning:** TensorFlow, Keras
- **Machine Learning:** scikit-learn, XGBoost, imbalanced-learn (SMOTE)
- **Data Handling:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Computer Vision:** OpenCV
- **NLP:** NLTK, Keras Tokenizer

## 📂 Repository Structure

```
├── DL_P1_Distracted_driver_multiaction_classification.ipynb
├── DL_P2_Multiclass_Multilabel_prediction_For_stack_overflow_Questions.ipynb
├── ML_P1_Consumer_Complaints_Resolution.ipynb
├── ML_P3_Counterfeit_Medicines_Sales_Prediction.ipynb
└── README.md
```

## ▶️ How to Run

1. Clone the repository
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```
2. Install dependencies
   ```bash
   pip install numpy pandas scikit-learn xgboost imbalanced-learn tensorflow keras opencv-python matplotlib seaborn nltk
   ```
3. Open any notebook in Jupyter or Google Colab and run the cells in order. Each notebook expects its dataset (linked/described at the top of the file) to be available in the working directory or, for the CV project, mounted from Kaggle input.

## 📊 Key Results

| Project | Metric | Result |
|---------|--------|--------|
| Distracted Driver Classification | Validation Accuracy / Classification Report | See notebook (confusion matrix + per-class report) |
| Stack Overflow Tag Prediction | F1-score (samples avg, threshold 0.55) | See notebook output |
| Consumer Complaints Resolution | ROC-AUC (Logistic Regression, SMOTE-balanced) | ~0.63 |
| Counterfeit Medicines Sales | Mean Absolute Error (tuned Decision Tree / Random Forest) | See notebook output |

## 👤 About

These notebooks were built as part of my Machine Learning and Deep Learning portfolio, demonstrating end-to-end workflows: data cleaning, feature engineering, model building, hyperparameter tuning, and evaluation across classification, multi-label, and regression problems.

Feel free to explore, fork, or reach out with questions!
