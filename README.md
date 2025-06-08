# 🏥 Medical Insurance Cost Prediction

This project predicts individual medical insurance costs based on demographic and lifestyle features. It was built as a beginner-friendly, project-based learning exercise to understand the end-to-end machine learning workflow — including building a linear regression model **from scratch** using NumPy.

---

## 🔍 Problem Statement

Healthcare costs are highly variable and influenced by many personal factors. The goal of this project is to use regression techniques to **predict insurance charges** based on:
- Age
- Sex
- BMI
- Number of Children
- Smoking status
- Region of residencex

---

## 📦 Dataset

- **Source**: [Kaggle – Medical Insurance Cost Prediction](https://www.kaggle.com/datasets/rahulvyasm/medical-insurance-cost-prediction)
- **Records**: ~1,300
- **Features**:
    - `age`, `sex`, `bmi`, `children`, `smoker`, `region`
    - `charges` (target)

---

## 🔨 What I Did

### 1. 📊 Exploratory Data Analysis (EDA)
- Analyzed feature distributions, outliers, and target variable behavior
- Used boxplots, histograms, and scatterplots to explore relationships
- Identified key drivers of medical cost (e.g., smoking status, age)

### 2. 🧹 Data Preprocessing
- Encoded categorical features (`sex`, `smoker`, `region`) using mapping and one-hot encoding
- Split the dataset into training and test sets
- Standardized numerical features (optional for certain models)

### 3. 🧠 Linear Regression (Built From Scratch)
- Defined prediction logic: `y = Xw + b`
- Computed loss using Mean Squared Error (MSE)
- Derived gradients manually and updated weights using gradient descent
- Tuned learning rate and epochs to observe convergence behavior

### 4. 📈 Model Evaluation
- Evaluated model using:
    - Mean Absolute Error (MAE)
    - Root Mean Squared Error (RMSE)
    - R² score (coefficient of determination)
- Visualized prediction vs. actual comparison

---

## 🚀 Results & Insights

- Manual linear regression achieved decent predictive performance
- Smoking status had the strongest impact on charges
- Age and BMI showed positive correlation with costs
- One-hot encoding was critical for proper handling of `region`

---

## 💡 Key Learnings

- Understood how weights and bias contribute to predictions
- Experienced the effect of learning rate and gradient stability
- Reinforced the math behind backpropagation and optimization
- Learned to interpret model output, residuals, and performance metrics

---

## 🛠 Tech Stack

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Jupyter Notebook

> 🔍 No `scikit-learn` was used for training the model — this was done fully from scratch to deepen understanding.

---

## 📁 Project Structure

```
insurance-cost-prediction/
├── data/
│   └── insurance.csv
├── notebooks/
│   ├── 01-data-exploration.ipynb
│   └── 02-preprocessing-and-modeling.ipynb
├── README.md
└── requirements.txt
```

---

## 📌 Next Steps (Stretch Goals)

- Compare with scikit-learn’s implementation of Linear Regression
- Try regularized models like Ridge and Lasso
- Implement a simple Neural Network from scratch
- Deploy model with Streamlit or Flask for real-time predictions

---

## 📚 Acknowledgments

- [Dataset by Rahul Vyas on Kaggle](https://www.kaggle.com/datasets/rahulvyasm/medical-insurance-cost-prediction)
