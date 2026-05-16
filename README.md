# 🚜 Bulldozer Sale Price Prediction using Machine Learning

## 📌 Overview

This project focuses on predicting bulldozer sale prices using machine learning regression techniques.  
The dataset is sourced from the Kaggle **Bluebook for Bulldozers** competition and contains historical auction data, machine specifications, and operational attributes.

The goal is to build a robust predictive model that estimates fair market prices for heavy machinery using structured tabular data.

### 🎯 Key Objectives
- Data preprocessing and cleaning
- Feature engineering from time-series auction data
- Handling missing values and high-cardinality categorical variables
- Exploratory Data Analysis (EDA)
- Training a Random Forest Regression model
- Hyperparameter optimization
- Model evaluation using RMSLE (Root Mean Squared Log Error)

---

## 📊 Dataset Information

- **Source:** Kaggle – Bluebook for Bulldozers
- **Rows:** 412,698
- **Initial Features:** 52
- **Engineered Features:** 59
- **Target Variable:** `SalePrice`

### 🧾 Example Features
- YearMade
- ProductSize
- MachineHoursCurrentMeter
- ProductGroup
- fiModelDesc
- Hydraulics
- Differential_Type
- sale_year, sale_month, sale_day

---

## 🛠️ Technologies & Libraries

### Programming Language
- Python

### Data Processing
- Pandas
- NumPy

### Visualization
- Matplotlib
- Seaborn

### Machine Learning
- Scikit-learn

---

## 🧹 Data Preprocessing & Feature Engineering

### 📅 Time-based Feature Engineering
Extracted from `saledate`:
- Sale Year
- Sale Month
- Sale Day
- Day of Week
- Day of Year

### 🔢 Categorical Encoding
- Converted object columns into categorical dtype
- Encoded categories into numeric codes for ML compatibility

### ❗ Missing Value Handling
- Median imputation for numerical variables
- Added binary indicators for imputed values

### 🧼 Data Cleaning
- Fixed invalid `YearMade` values
- Removed raw datetime column after feature extraction

---

## 📈 Exploratory Data Analysis (EDA)

Key analyses performed:
- Distribution of `SalePrice` over time
- Correlation heatmaps
- Feature importance analysis
- Residual error distribution
- Actual vs Predicted comparisons

### 🔍 Key Insights
- `YearMade` and `ProductSize` are strong predictors of price
- High-value machines exhibit higher prediction variance
- Dataset contains significant missingness and high-cardinality categorical variables
- Sale prices show right-skewed distribution

---

## 🤖 Machine Learning Model

### 🌲 Random Forest Regressor

A Random Forest model was selected due to its:
- Robustness to non-linear relationships
- Strong performance on tabular data
- Ability to handle mixed feature types

```python
RandomForestRegressor()
```

---

## 📊 Initial Model Performance

| Metric | Train | Test |
|--------|------|------|
| R² Score | 0.9874 | 0.8588 |
| MAE | 1568.72 | 5921.48 |
| RMSLE | 0.0848 | 0.2447 |

⚠️ Observation: Initial model showed signs of overfitting.

---

## 🔧 Hyperparameter Optimization

Optimized using `RandomizedSearchCV`.

### Best Parameters
```python
{
    "max_depth": None,
    "max_features": 0.5,
    "max_samples": 22068,
    "min_samples_split": 10,
    "n_estimators": 300
}
```

---

## 📉 Log Transformation

To improve model stability and RMSLE performance:

```python
np.log1p(SalePrice)
```

Predictions were transformed back using:

```python
np.expm1(predictions)
```

### 📌 Impact
- Reduced variance in target distribution
- Improved RMSLE significantly

---

## 🏁 Final Model Performance

| Metric | Train | Test |
|--------|------|------|
| R² Score | 0.9347 | 0.8480 |
| MAE | 3513.77 | 5994.53 |
| RMSLE | 0.0151 | 0.0216 |

---

## 📌 Feature Importance

Top influential features:
1. YearMade  
2. ProductSize  
3. Enclosure  
4. fiSecondaryDesc  
5. sale_year  

### 🧠 Interpretation
- Newer machines tend to have higher prices
- Machine size strongly correlates with operational value
- Configuration and equipment type significantly influence auction price

---

## 📊 Model Evaluation Visuals

- Correlation Heatmap
- Feature Importance Plot
- Actual vs Predicted Scatter Plot
- Residual Distribution Plot
- SalePrice distribution over time

---

## 📁 Project Structure

```bash
Bulldozer-Price-Prediction/
│
├── data/          # Raw dataset (Kaggle Bulldozers dataset)
├── notebooks/     # EDA and model development notebooks
├── images/        # Visualizations and plots
├── models/        # Trained ML models (.pkl files)
├── README.md      # Project documentation
└── requirements.txt
```

---

## 📌 Business Impact

This model can support:
- Auction houses in fair price estimation
- Equipment valuation systems
- Market pricing strategy optimization
- Reduction of pricing inefficiencies in heavy machinery markets

---

## 🚀 Future Improvements

- Implementation of Gradient Boosting models (XGBoost, LightGBM)
- Advanced feature engineering (interaction features)
- Automated ML pipeline (sklearn pipelines / MLflow)
- Model deployment using Flask or Streamlit
- Real-time price prediction API

---

## 👨‍💻 Author

**Sami Sevdi**  
Industrial Engineer | Data Scientist | Machine Learning Specialist    

---

## 📄 License

This project is intended for educational and research purposes.
