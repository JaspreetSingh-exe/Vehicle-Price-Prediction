# 🚗 Vehicle Price Prediction

## 📌 Problem Statement
The goal of this project is to develop a machine learning model that can accurately predict the price of a vehicle based on various features such as make, model, year, fuel type, transmission, mileage, and more. The dataset contains numerical and categorical features, which require preprocessing before training the model.

---

## 📂 Dataset Description

The dataset used for vehicle price prediction consists of **17 columns** with **1002 entries**. Below is the detailed description of each column:

| #  | Column Name      | Non-Null Count | Data Type | Description |
|----|----------------|----------------|-----------|-------------|
| 0  | name           | 1002 non-null  | object    | Name of the vehicle listing |
| 1  | description    | 946 non-null   | object    | Description of the vehicle |
| 2  | make          | 1002 non-null  | object    | Manufacturer of the vehicle |
| 3  | model         | 1002 non-null  | object    | Model name of the vehicle |
| 4  | year          | 1002 non-null  | int64     | Manufacturing year of the vehicle |
| 5  | price         | 979 non-null   | float64   | Price of the vehicle (Target variable) |
| 6  | engine        | 1000 non-null  | object    | Engine type of the vehicle |
| 7  | cylinders     | 897 non-null   | float64   | Number of cylinders in the engine |
| 8  | fuel          | 995 non-null   | object    | Type of fuel used |
| 9  | mileage       | 968 non-null   | float64   | Mileage of the vehicle (in miles per gallon) |
| 10 | transmission  | 1000 non-null  | object    | Type of transmission (Automatic/Manual) |
| 11 | trim          | 1001 non-null  | object    | Specific trim/version of the vehicle model |
| 12 | body          | 999 non-null   | object    | Body type of the vehicle (SUV, Sedan, etc.) |
| 13 | doors         | 995 non-null   | float64   | Number of doors in the vehicle |
| 14 | exterior_color| 997 non-null   | object    | Color of the vehicle's exterior |
| 15 | interior_color| 964 non-null   | object    | Color of the vehicle's interior |
| 16 | drivetrain    | 1002 non-null  | object    | Type of drivetrain (FWD, AWD, etc.) |

The **target variable** for our prediction task is **`price`**, which we aim to predict based on the other vehicle attributes.


## 💡 Solution Approach
This project follows a structured approach to ensure efficient data handling and model training. Below is an overview of the key steps:

1. **Project Setup & Installation**
2. **Exploratory Data Analysis (EDA)**
3. **Feature Engineering & Encoding**
4. **Scaling Numerical Features**
5. **Splitting Data into Train & Test Sets**
6. **Training Various Machine Learning Models**
7. **Hyperparameter Tuning using GridSearchCV**
8. **Evaluating & Comparing Model Performance**

---


## 📂 Project Structure

The following is the directory structure of the Vehicle Price Prediction project:

```
Vehicle Price Prediction/
|-- .idea/                              # IDE Configuration Files (Optional)
|-- catboost_info/                      # CatBoost Model Training Logs
|   |-- learn/                          # Learning Data
|   |-- tmp/                            # Temporary Files
|   |-- catboost_training.json          # CatBoost Training Metadata
|   |-- learn_error.tsv                 # CatBoost Learning Error Log
|   |-- time_left.tsv                   # Remaining Training Time
|-- dataset.csv                         # Vehicle Dataset
|-- Predict Vehicle Prices.pdf          # Project PDF
|-- price_prediction.ipynb              # Jupyter Notebook with Code & Analysis
|-- README.md                           # Project README File
|-- requirements.txt                    # Dependencies Required for the Project
|-- LICENSE                             # License File
```


## 🛠️ Installation
To run this project locally, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo-name.git
   ```

2. Navigate to the project directory:
   ```sh
   cd vehicle-price-prediction
   ```

3. Create and activate a virtual environment (optional but recommended):
   ```sh
   python -m venv venv
   source venv/bin/activate  # On MacOS/Linux
   venv\Scripts\activate     # On Windows
   ```

4. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

5. Run the Jupyter Notebook:
   ```sh
   jupyter price_prediction
   ```

---

## 📊 Exploratory Data Analysis (EDA)
EDA helps in understanding the dataset by analyzing distributions, relationships between features, and identifying potential data quality issues.

### Checking for Missing Values
```python
missing_values = vehicle_data.isnull().sum()
print(missing_values[missing_values > 0])
```

### Visualizing Numerical Features
```python
import seaborn as sns
import matplotlib.pyplot as plt
sns.histplot(vehicle_data['mileage'], kde=True)
plt.title('Mileage Distribution')
plt.show()
```

### Visualizing Categorical Features
```python
sns.countplot(x=vehicle_data['fuel'])
plt.title('Fuel Type Distribution')
plt.show()
```

### Checking Correlation Between Features
```python
import seaborn as sns
plt.figure(figsize=(10,6))
sns.heatmap(vehicle_data.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()
```

---

## 📊 Data Preprocessing
Data preprocessing is a crucial step where we prepare the data for model training by handling missing values, removing outliers, encoding categorical variables, and scaling numerical features.

### Steps in Data Preprocessing:
1. **Handling Missing Values:** Dropped missing values to avoid inconsistencies.
2. **Outlier Detection & Removal:** Used the IQR method to identify and remove extreme values.
3. **Encoding Categorical Features:** Applied Label Encoding for high-cardinality categorical features and One-Hot Encoding for others.
4. **Scaling Numerical Features:** Used StandardScaler to normalize numerical variables.

### Handling Outliers Using IQR
```python
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]
```

### Encoding Categorical Variables
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
refined_data['make'] = le.fit_transform(refined_data['make'])
refined_data['model'] = le.fit_transform(refined_data['model'])
```

### Scaling Numerical Features
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_features = ['year', 'cylinders', 'mileage', 'doors']
refined_data[scaled_features] = scaler.fit_transform(refined_data[scaled_features])
```

---

## 🚀 Model Training
Model training involves splitting the dataset into training and testing sets, training various machine learning models, and evaluating their performance.

### Splitting Data
```python
from sklearn.model_selection import train_test_split
X = refined_data.drop('price', axis=1)
y = refined_data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Linear Regression Model
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### Random Forest Model
```python
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
```

### Gradient Boosting Model
```python
from sklearn.ensemble import GradientBoostingRegressor
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
```

### CatBoost Model
```python
from catboost import CatBoostRegressor
cat_model = CatBoostRegressor(iterations=500, learning_rate=0.1, depth=6, verbose=False, random_state=42)
cat_model.fit(X_train, y_train)
cat_pred = cat_model.predict(X_test)
```

### Stacking Regressor
```python
from sklearn.ensemble import StackingRegressor
stack_model = StackingRegressor(
    estimators=[
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42))
    ],
    final_estimator=LinearRegression()
)
stack_model.fit(X_train, y_train)
stack_pred = stack_model.predict(X_test)
```

---

## 📈 Model Performance Comparison

| Model                   | R² Score | RMSE    |
|-------------------------|----------|---------|
| Tuned Gradient Boosting | 0.8664   | 6061.99 |
| CatBoost                | 0.8634   | 6130.54 |
| Gradient Boosting       | 0.8612   | 6178.28 |
| Stacking Regressor      | 0.8616   | 6180.65 |
| Random Forest           | 0.8411   | 6611.37 |
| Linear Regression       | 0.6999   | 9086.25 |

---  

## 🔍 Key Takeaways:
✅ **Tuned Gradient Boosting** emerged as the best model, achieving the highest **R² Score (0.8664)** and the lowest **RMSE (6061.99)**, making it the most accurate predictor of vehicle prices.

✅ **CatBoost and Gradient Boosting** also delivered strong results, confirming that ensemble learning techniques are highly effective for this task.

✅ **Stacking Regressor**, which combines multiple models, performed almost as well as individual boosting models but didn’t outperform the tuned gradient boosting.

✅ **Random Forest** provided good performance but lagged behind boosting models, indicating that gradient boosting is more suitable for this type of structured data.    

✅ **Linear Regression**, a simpler model, had the lowest accuracy. This suggests that **vehicle pricing is a complex problem requiring non-linear models** to capture interactions between features.      

---  

## 📌 Conclusion
- We built a vehicle price prediction model by following a structured machine learning pipeline.
- Several regression models were tested, and **Gradient Boosting** with hyperparameter tuning performed the best with an **R² Score of 0.8664** and **RMSE of 6061.99**.
- If computational efficiency is a concern, **CatBoost** provides a great balance between performance and speed.
- **Stacking models** can be explored further for possible performance improvements. 

Future improvements can include:
- Trying more feature engineering techniques
- Experimenting with deep learning models
- Deploying the model using Flask or FastAPI

## 🤝 Contribution

We welcome contributions from the community! If you would like to improve this project, feel free to:
- **Fork the repository** 🍴
- **Make enhancements** 🔧
- **Fix issues and bugs** 🐞
- **Optimize model performance** 📊
- **Suggest new features** 🚀

If you find any **bugs or issues**, please raise them in the **[Issues section](https://github.com/JaspreetSingh-exe/Vehicle-Price-Prediction/issues)** of this repository.

---

## 📞 Contact

If you have any questions or want to collaborate, feel free to reach out:

📧 **Email**: [jaspreetsingh01110@gmail.com](mailto:jaspreetsingh01110@gmail.com)  

---

## 📜 License
This project is licensed under the MIT License.

---

💙 If you found this project helpful, consider giving it a ⭐ on GitHub!

