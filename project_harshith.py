#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def load_data(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data

def process_data(df):
    df['Title'] = df['Name'].apply(extract_title)
    df['Deck'] = df['Cabin'].apply(lambda x: str(x)[0] if pd.notnull(x) else 'Unknown')
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # Bin ages into categories
    bins_age = [-1, 18, 30, 50, float('inf')]
    labels_age = ['children', 'youth', 'middleaged', 'old']
    df['AgeCategory'] = pd.cut(df['Age'], bins=bins_age, labels=labels_age)

    # Create Family_Size column
    df['Family_Size'] = df['SibSp'] + df['Parch'] + 1

    # Bin family size into categories
    bins_family_size = [0, 1, 4, 6, float('inf')]
    labels_family_size = ['Alone', 'Small', 'Medium', 'Large']
    df['Family_Size_Category'] = pd.cut(df['Family_Size'], bins=bins_family_size, labels=labels_family_size)

    # One-hot encode columns
    columns_to_encode = ['Embarked', 'Title', 'Deck', 'Pclass', 'AgeCategory', 'Family_Size_Category']
    df = one_hot_encode(df, columns_to_encode)

    columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Age', 'SibSp', 'Parch', 'Family_Size']
    df.drop(columns=columns_to_drop, axis=1, inplace=True)

    df.fillna(df.mean(), inplace=True)
    return df

def save_predictions_to_csv(passenger_ids, predictions, model_name):
    result_df = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': predictions})
    result_df.to_csv(f'{model_name}_predictions.csv', index=False)
    print(f'{model_name} predictions saved to {model_name}_predictions.csv')

def one_hot_encode(df, columns):
    return pd.get_dummies(df, columns=columns, drop_first=True)

def extract_title(name):
    title_mapping = {
        "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
        "Dr": "Other", "Rev": "Other", "Col": "Other", "Major": "Other",
        "Mlle": "Miss", "Ms": "Miss", "Don": "Other", "Dona": "Other",
        "Mme": "Mrs", "Lady": "Other", "Sir": "Other", "Capt": "Other",
        "Countess": "Other", "Jonkheer": "Other"
    }
    title = name.split(',')[1].split('.')[0].strip()
    return title_mapping.get(title, "Other")

def split_data(df):
    X = df.drop('Survived', axis=1)
    y = df['Survived'].values
    return train_test_split(X, y, test_size=0.2, random_state=42)

def fill_missing_features(X_train, X_test):
    missing_features = set(X_train.columns) - set(X_test.columns)
    for feature in missing_features:
        X_test[feature] = 0
    return X_test

def train_and_evaluate_models_with_kfold(X_train, y_train, models, n_splits_values):
    results = {}

    for n_splits in n_splits_values:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        for model_name, (model_class, param_grid) in models.items():
            print(f"\nTraining {model_name} with K-Fold (n_splits={n_splits})...")
            model = model_class()  # Create an instance of the model
            best_params = {}
            best_accuracy = 0

            for train_index, cv_index in kf.split(X_train):
                X_kf_train, X_kf_cv = X_train.iloc[train_index], X_train.iloc[cv_index]
                y_kf_train, y_kf_cv = y_train[train_index], y_train[cv_index]

                if param_grid:
                    print(f"Performing RandomizedSearchCV for {model_name}...")
                    random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=10, cv=5, scoring='accuracy', random_state=42)
                    random_search.fit(X_kf_train, y_kf_train)

                    if random_search.best_score_ > best_accuracy:
                        best_params = random_search.best_params_
                        best_accuracy = random_search.best_score_
                else:
                    best_params = {}

            final_model = model_class(**best_params)

            final_model.fit(X_train, y_train)

            preds_cv = final_model.predict(X_train)

            accuracy_cv = accuracy_score(y_train, preds_cv)

            report_cv = classification_report(y_train, preds_cv)

            results[model_name] = {'model': final_model, 'accuracy_cv': accuracy_cv, 'classification_report_cv': report_cv}
            print(f"{model_name} training complete with K-Fold (n_splits={n_splits}). Best hyperparameters: {best_params}")

    return results

def print_cross_val_results(results):
    for model_name, result in results.items():
        print(f"\n{model_name} Results:")
        print(f"Cross-Validation Accuracy: {result['accuracy_cv']}")
        print(f"Classification Report on Cross-Validation Set:\n{result['classification_report_cv']}")

def plot_metrics(results):
    model_names = list(results.keys())
    accuracy_scores = []
    precision_scores = []
    f1_scores = []

    for model_name, result in results.items():
        report_lines = result['classification_report_cv'].split('\n')
        accuracy_scores.append(result['accuracy_cv'])
        precision_scores.append(float(report_lines[3].split()[1]))
        f1_scores.append(float(report_lines[2].split()[3]))

    x = np.arange(len(model_names))
    width = 0.2

    # Use Seaborn color palette for better visualization
    colors = sns.color_palette('Set3', len(model_names))

    fig, ax = plt.subplots(figsize=(10, 6))

    rects1 = ax.bar(x - width, accuracy_scores, width, label='Accuracy', color=colors[0])
    rects2 = ax.bar(x, precision_scores, width, label='Precision', color=colors[1])
    rects3 = ax.bar(x + width, f1_scores, width, label='F1 Score', color=colors[2])

    ax.set_ylabel('Scores')
    ax.set_title('Model Evaluation Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(round(height, 2)),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    fig.tight_layout()
    plt.show()
    
    
def plot_conf_matrix(y_true, y_pred, title):
    cm = [[0, 0], [0, 0]]  # Initialize a 2x2 confusion matrix

    for true, pred in zip(y_true, y_pred):
        cm[true][pred] += 1

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()

    classes = ['Not Survived', 'Survived']
    tick_marks = range(len(classes))

    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i][j]), ha='center', va='center', color='red' if cm[i][j] > cm[1][1] else 'black')

    plt.show()

def main():
    train_data, test_data1 = load_data('./train.csv', './test.csv')

    train_data = process_data(train_data)
    test_data = process_data(test_data1)

    print(train_data.head())

    X_train, X_cv, y_train, y_cv = split_data(train_data)
    
    X_test = test_data
    X_test = fill_missing_features(X_train, X_test)

    # Reorder columns in X_test to match the order in X_train
    X_test = X_test[X_train.columns]

    models = {
        'SVM': (SVC, {'C': [0.1, 1, 10, 100], 'gamma': [0.1, 0.01, 0.001, 0.0001]}),
        'Neural Network': (MLPClassifier, {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'alpha': [0.0001, 0.001, 0.01, 0.1],
                                           'max_iter': [100, 200, 300], 'activation': ['relu', 'tanh'], 'solver': ['sgd', 'adam']}),
        'Random Forest': (RandomForestClassifier, {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]})
    }

    n_splits_values = [5, 10]

    results = train_and_evaluate_models_with_kfold(X_train, y_train, models, n_splits_values)

    print_cross_val_results(results)

    for model_name, result in results.items():
        preds_test = result['model'].predict(X_test)
        print(f"\n{model_name} Test Set Predictions:")
        print(preds_test)
        
        # Save predictions to CSV
        save_predictions_to_csv(test_data1['PassengerId'], preds_test, model_name)
        

    plot_conf_matrix(y_cv, results['SVM']['model'].predict(X_cv), 'Confusion Matrix - SVM')
    plot_conf_matrix(y_cv, results['Neural Network']['model'].predict(X_cv), 'Confusion Matrix - Neural Network')
    plot_conf_matrix(y_cv, results['Random Forest']['model'].predict(X_cv), 'Confusion Matrix - Random Forest')

    plot_metrics(results)

if __name__ == "__main__":
    main()


# In[ ]:




