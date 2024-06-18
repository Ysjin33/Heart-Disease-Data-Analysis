## Import common modules for data analysis ##
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Preprocess the data."""
    df = df.dropna()  # Handle missing values
    age_bins = [0, 20, 40, 60, 100]
    age_labels = ['Youth', 'Young Adult', 'Middle-aged adult', 'Old']
    df['Age_Cat'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)
    return df

def plot_data(df):
    """Generate plots for data analysis."""
    plt.figure()
    df['HeartDisease'].value_counts(normalize=True).plot(kind='bar')
    plt.title('Heart Disease Distribution')
    plt.show()

    sns.catplot(data=df, y='HeartDisease', x='Sex', kind='bar')
    plt.title('Male and Female Ratio for Heart Disease')
    plt.show()

    plt.figure(figsize=(30, 20))
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist() + ['Age_Cat']
    for i, col in enumerate(categorical_cols):
        plt.subplot(3, 5, i + 1)
        sns.countplot(x=col, hue='HeartDisease', data=df)
        plt.title(col)
    plt.tight_layout()
    plt.show()

    int_cols = df.select_dtypes(['int64'])
    plt.figure(figsize=(12, 7))
    sns.heatmap(int_cols.corr(), annot=True)
    plt.title('Correlation Heatmap')
    plt.show()

    plt.figure(figsize=(20, 10))
    int_cols.boxplot(grid=False)
    plt.title('Boxplot of Continuous Variables')
    plt.show()

    g = sns.FacetGrid(df, row='Sex', col='Age_Cat')
    g.map(sns.histplot, "HeartDisease")
    plt.show()

    plt.figure(figsize=(15, 10))
    sns.boxplot(data=df, x="Age_Cat", y="Cholesterol")
    plt.title('Boxplot of Cholesterol by Age Category')
    plt.show()

    plt.figure(figsize=(30, 20))
    for i in enumerate(int_cols.columns):
        plt.subplot(6, 3, i[0]+1)
        sns.boxplot(x=i[1], data=int_cols)
        plt.title(f'Boxplot of {i[1]}')
    plt.tight_layout()
    plt.show()

def oversample_data(X, y):
    """Apply SMOTE to balance the dataset."""
    os = SMOTE(random_state=0)
    X_os, y_os = os.fit_resample(X, y)
    return X_os, y_os

def build_pipeline():
    """Build a machine learning pipeline."""
    numerical_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    categorical_features = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope', 'Age_Cat']
    
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)])

    model = LogisticRegression(solver='liblinear')

    pipeline = ImbPipeline(steps=[('preprocessor', preprocessor),
                                  ('smote', SMOTE(random_state=0)),
                                  ('model', model)])
    return pipeline

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and generate performance metrics."""
    pred_y = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred_y)
    print("Test Accuracy: {:.2f}%".format(accuracy * 100))

    matrix = confusion_matrix(y_test, pred_y)
    sns.heatmap(matrix / np.sum(matrix), annot=True, fmt='.2%', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()

    report = classification_report(y_test, pred_y)
    print("Classification Report:")
    print(report)

    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print("ROC AUC Score: {:.2f}".format(roc_auc))

    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, marker='.')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

    precision, recall, _ = precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])
    plt.figure()
    plt.plot(recall, precision, marker='.')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()

def cross_validate_model(pipeline, X, y):
    """Perform cross-validation and print results."""
    kfold = KFold(n_splits=10, random_state=7, shuffle=True)

    results_acc = cross_val_score(pipeline, X, y, cv=kfold)
    print("Cross-validated Accuracy: %.3f (%.3f)" % (results_acc.mean(), results_acc.std()))

    results_log_loss = cross_val_score(pipeline, X, y, cv=kfold, scoring='neg_log_loss')
    print("Log Loss: %.3f (%.3f)" % (results_log_loss.mean(), results_log_loss.std()))

    results_auc = cross_val_score(pipeline, X, y, cv=kfold, scoring='roc_auc')
    print("AUC: %.3f (%.3f)" % (results_auc.mean(), results_auc.std()))

def save_model(pipeline, model_path):
    """Save the trained model to a file."""
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")

def main():
    data_file_path = 'heart.csv'  
    model_path = 'heart_disease_model.pkl'  

    df = load_data(data_file_path)

    print("Original Data:")
    print(df.head())

    df = preprocess_data(df)
    plot_data(df)

    X = df.drop(columns='HeartDisease')
    y = df['HeartDisease']

    pipeline = build_pipeline()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    pipeline.fit(X_train, y_train)

    evaluate_model(pipeline, X_test, y_test)
    cross_validate_model(pipeline, X_train, y_train)
    save_model(pipeline, model_path)

if __name__ == "__main__":
    main()