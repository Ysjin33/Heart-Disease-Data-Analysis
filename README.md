# Heart Disease Prediction Model

This repository contains a machine learning project to predict heart disease using health metrics. The code includes data preprocessing, exploratory data analysis, model building, evaluation, and saving the trained model.

## Project Steps

1. **Data Loading**: Load the dataset from a CSV file.
2. **Data Preprocessing**: Handle missing values and create new categorical features.
3. **Exploratory Data Analysis (EDA)**: Visualize data distribution and relationships.
4. **Data Balancing**: Use SMOTE to balance the dataset.
5. **Model Building**: Create a machine learning pipeline and train a logistic regression model.
6. **Model Evaluation**: Evaluate the model using various metrics and cross-validation.
7. **Model Saving**: Save the trained model for future use.

## Requirements

Install the required Python packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib
```

## Usage

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/yourusername/heart-disease-prediction.git
    cd heart-disease-prediction
    ```

2. **Prepare the Data**:

    Place `heart.csv` in the repository directory.

3. **Run the Script**:

    Execute the main script:

    ```bash
    python main.py
    ```

## Main Functions

### Data Loading

```python
def load_data(file_path):
    return pd.read_csv(file_path)
```

### Data Preprocessing

```python
def preprocess_data(df):
    df = df.dropna()
    age_bins = [0, 20, 40, 60, 100]
    age_labels = ['Youth', 'Young Adult', 'Middle-aged adult', 'Old']
    df['Age_Cat'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)
    return df
```

### Data Visualization

```python
def plot_data(df):
    # Various plots for EDA
```

### Data Balancing

```python
def oversample_data(X, y):
    os = SMOTE(random_state=0)
    X_os, y_os = os.fit_resample(X, y)
    return X_os, y_os
```

### Model Pipeline

```python
def build_pipeline():
    # Pipeline construction
```

### Model Evaluation

```python
def evaluate_model(model, X_test, y_test):
    # Evaluation metrics and plots
```

### Cross-Validation

```python
def cross_validate_model(pipeline, X, y):
    # Cross-validation results
```

### Model Saving

```python
def save_model(pipeline, model_path):
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
