# Importing necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, KFold
import random
import warnings
warnings.filterwarnings('ignore')

# Setting the data path
datapath = "./"

# Reading train and test datasets
traincsvname = datapath + 'train.csv'
testcsvname = datapath + 'test.csv'
df_train = pd.read_csv(traincsvname)
df_test = pd.read_csv(testcsvname)

# Exploratory Data Analysis function
def exploratory_data_analysis(df_train, df_test):
    # Displaying basic information about the training dataset
    df_train.head()
    df_train.shape
    df_train.info()
    df_train.describe().T
    
    # Displaying basic information about the testing dataset
    df_test.head()
    df_test.shape
    df_test.info()
    df_test.describe().T

    # Separating variable types
    numerical_variables = [feature for feature in df_train.columns if df_train[feature].dtype in ['int_', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16',
                                   'uint32', 'uint64','float_', 'float16', 'float32','float64']]
    print('Number of numerical variables =>',len(numerical_variables),'\nNumerical Variables=>',numerical_variables)

    continuous_variables =[feature for feature in numerical_variables if len(df_train[feature].unique()) > 25] 
    print('Number of continuous variables =>',len(continuous_variables),'\nContinuous Variables=>',continuous_variables)

    discrete_variables =[feature for feature in numerical_variables if len(df_train[feature].unique()) < 25] 
    print('Number of discrete variables =>',len(discrete_variables),'\nDiscrete Variables=>',discrete_variables)

    categorical_variables = [feature for feature in df_train.columns if df_train[feature].dtype in ['O','bool_']]
    print('Number of categorical variables =>',len(categorical_variables),'\nCategorical Variables=>',categorical_variables)

    # Displaying the number of categories in categorical variables
    total=0
    for feature in categorical_variables:
        print(feature,'=>',df_train[feature].nunique())
        total += df_train[feature].nunique()
    print('Total category:',total)

    # Checking for missing values in the training dataset
    df_train.isnull().sum()

    print("In train columns")
    for column in df_train.columns:
        print('Percentage of null in',column,':',df_train[column].isnull().mean()*100)

    # Checking for missing values in the testing dataset
    df_test.isnull().sum()

    print("In test columns")
    for column in df_test.columns:
        print('Percentage of null in',column,':',df_test[column].isnull().mean()*100)

    # Visualizing missing values in the training dataset
    sns.heatmap(df_train.isnull(),linecolor='red')

    # Visualizing a distribution plot for the 'Sex' feature in relation to 'Survived'
    df_train_dist = df_train.copy()
    df_train_dist['Survived'] = df_train_dist['Survived'].astype('O')
    sns.displot(df_train,x=df_train['Sex'], hue='Survived', bins=40)

    # Handling duplicates in the training dataset
    df_train.duplicated().sum()
    df_train['PassengerId'].duplicated().sum()
    df_train['Name'].duplicated().sum()
    df_train['Ticket'].duplicated().sum()
    df_train['Ticket'].duplicated().mean()

    # Extracting titles from 'Name' and checking unique values
    df_train['Name'].str.split(',').str[1].str.split('.').str[0].unique()
    df_train['Name'].str.split(',').str[1].str.split('.').str[0].nunique()

# Commented the below line as it is not required to run every time
# exploratory_data_analysis(df_train, df_test)

# Function to handle missing values
def handle_missing_values(df_train, df_test):
    knn_impute = KNNImputer(n_neighbors=5)
    df_train[['Age','Survived']] = knn_impute.fit_transform(df_train[['Age','Survived']])
    df_test[['Age','Fare']] = knn_impute.fit_transform(df_test[['Age','Fare']])

    # Only 2 rows have null 'Embarked', so filled with mode
    df_train.fillna(value={'Embarked': df_train['Embarked'].mode()[0]}, inplace=True)

    return df_train, df_test

df_train, df_test = handle_missing_values(df_train, df_test)

# Function to handle imbalanced dataset
def handling_imbalanced_dataset(df_train):
    print(f"Percentage of not survived {df_train['Survived'].value_counts()[0] / df_train['Survived'].value_counts().sum()}")
    print(f"Percentage of survived {df_train['Survived'].value_counts()[1] / df_train['Survived'].value_counts().sum()}")

# handling_imbalanced_dataset(df_train)

# Function to handle outliers
def handling_outliers(df_train, df_test):
    numerical_variables = [feature for feature in df_train.columns if df_train[feature].dtype in ['int_', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16',
                                   'uint32', 'uint64','float_', 'float16', 'float32','float64']]

    continuous_variables = [feature for feature in numerical_variables if len(df_train[feature].unique()) > 25] 
    
    # Removing outliers for 'Fare' and 'Age' in training data
    IQR = df_train.Fare.quantile(0.75) - df_train.Fare.quantile(0.25)
    lower_bridge = df_train['Fare'].quantile(0.25) - (IQR * 3)
    upper_bridge = df_train['Fare'].quantile(0.75) + (IQR * 3)
    df_train[df_train.Fare > upper_bridge]['Fare'] = upper_bridge

    IQR = df_train.Age.quantile(0.75) - df_train.Age.quantile(0.25)
    lower_bridge = df_train['Age'].quantile(0.25) - (IQR * 3)
    upper_bridge = df_train['Age'].quantile(0.75) + (IQR * 3)
    df_train[df_train.Age > upper_bridge]['Age'] = upper_bridge
    
    # Removing outliers for 'Fare' and 'Age' in testing data
    IQR = df_test.Fare.quantile(0.75) - df_test.Fare.quantile(0.25)
    lower_bridge = df_test['Fare'].quantile(0.25) - (IQR * 3)
    upper_bridge = df_test['Fare'].quantile(0.75) + (IQR * 3)
    df_test[df_test.Fare > upper_bridge]['Fare'] = upper_bridge

    IQR = df_test.Age.quantile(0.75) - df_test.Age.quantile(0.25)
    lower_bridge = df_test['Age'].quantile(0.25) - (IQR * 3)
    upper_bridge = df_test['Age'].quantile(0.75) + (IQR * 3)
    df_test[df_test.Age > upper_bridge]['Age'] = upper_bridge
    
    return df_train, df_test
    
df_train, df_test = handling_outliers(df_train, df_test)

# Separating features and target variable
X_train = df_train.drop('Survived', axis=1)
Y_train = df_train['Survived']
X_test = df_test.copy()

# Function for alternative one-hot encoding
def alternative_ohe(df, percentage, variables):
    df_ohe = df.copy()
    for feature in variables:     
        dicto = {feature: df_ohe[feature].value_counts(ascending=False).index,
                 'Value_counts': df_ohe[feature].value_counts(ascending=False).values}
        df_fe = pd.DataFrame(data=dicto)
        df_fe['Value_Ratio'] = df_fe['Value_counts'] / df_ohe.shape[0]
        
        total_perc = 0.0
        
        for i in range (0, len(df_fe)):
            if (total_perc <= (percentage / 100)):
                total_perc += df_fe.iloc[i:i+1]['Value_Ratio'].values[0]
            else:
                df_fe = df_fe.iloc[0:i]
                break
        for category in df_fe[feature].values:
            df_ohe[feature + '_' + category] = np.where(df_ohe[feature] == category, 1, 0)
    df_ohe.drop(variables, axis=1, inplace=True)
    return df_ohe

# Applying one-hot encoding to features
X_train_ohe = alternative_ohe(X_train, 100, ['Embarked', 'Sex'])
X_test_ohe = alternative_ohe(X_test, 100, ['Embarked', 'Sex'])

# Dropping unnecessary columns
X_train_ohe.drop(['PassengerId', 'Cabin', 'Ticket'], axis=1, inplace=True)
X_test_ohe.drop(['PassengerId', 'Cabin', 'Ticket'], axis=1, inplace=True)

# Extracting and processing title information from 'Name'
X_train_ohe['Title'] = X_train_ohe['Name'].str.split(',').str[1].str.split('.').str[0]
X_test_ohe['Title'] = X_test_ohe['Name'].str.split(',').str[1].str.split('.').str[0]

# Dropping 'Name' column
X_train_ohe.drop('Name', axis=1, inplace=True)
X_test_ohe.drop('Name', axis=1, inplace=True)

# Grouping titles
X_train_ohe['Title'] = X_train_ohe['Title'].replace([' Mlle', ' Ms'], ' Miss')
X_train_ohe['Title'] = X_train_ohe['Title'].replace(' Mme', ' Mrs')
X_train_ohe['Title'] = X_train_ohe['Title'].replace([' Dr', ' Rev', ' Major', ' Col', ' the Countess', ' Capt', ' Sir', ' Lady', ' Don', ' Jonkheer'], 'Other')

X_test_ohe['Title'] = X_test_ohe['Title'].replace([' Col', ' Rev', ' Dr', ' Dona'], 'Other')
X_test_ohe['Title'] = X_test_ohe['Title'].replace([' Mlle', ' Ms'], ' Miss')
X_test_ohe['Title'] = X_test_ohe['Title'].replace(' Mme', ' Mrs')

# Creating bins for 'Age' and 'Fare' features
data = [X_train_ohe, X_test_ohe]
for dataset in data:
    dataset['Age_bin'] = pd.cut(dataset['Age'], bins=[0, 12, 20, 40, 120], labels=['Children', 'Teenage', 'Adult', 'Elder'])
    dataset['Fare_bin'] = pd.cut(dataset['Fare'], bins=[0, 7.91, 14.45, 31, 120], labels=['Low_fare', 'median_fare', 'Average_fare', 'high_fare'])
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

# Dropping 'Age' and 'Fare' columns
for dataset in data:
    drop_column = ['Age', 'Fare']
    dataset.drop(drop_column, axis=1, inplace=True)

# Function to group family size
def family_group(size):
    a = ''
    if size <= 1:
        a = 'alone'
    elif size <= 4:
        a = 'small'
    else:
        a = 'large'
    return a

# Applying family grouping function
X_train_ohe['FamilyGroup'] = X_train_ohe.FamilySize.map(family_group)
X_test_ohe['FamilyGroup'] = X_test_ohe.FamilySize.map(family_group)

# Applying alternative one-hot encoding to additional features
X_train_ohe = alternative_ohe(X_train_ohe, 100, ['Title', 'Age_bin', 'Fare_bin', 'FamilyGroup'])
X_test_ohe = alternative_ohe(X_test_ohe, 100, ['Title', 'Age_bin', 'Fare_bin', 'FamilyGroup'])

# Aligning columns in the test set with those in the training set
X_test_ohe = X_test_ohe[X_train_ohe.columns]

def random_forest_modelling(X_train_ohe, Y_train, X_test):
    
    """
    Perform Random Forest classification, including hyperparameter tuning, training the model, and making predictions.

    Parameters:
    - X_train_ohe: Training data features after one-hot encoding.
    - Y_train: Training data labels.
    - X_test_ohe: Test data features.

    Output:
    - Generates and saves a confusion matrix plot for the training data.
    - Outputs a classification report and training accuracy for the Random Forest Classifier.
    - Saves predictions for the test data in 'random_forest.csv'.
    """

    # Assuming X_train_ohe and Y_train are your training data
    # The below commented code is for k fold cross validation
#     param_grid = {
#         'n_estimators': [100, 200, 300],
#         'max_depth': [None, 10, 20],
#         'min_samples_split': [1, 2, 5],
#         'min_samples_leaf': [1, 2]
#     }

#     random.seed(42)
    
#     random_forest = RandomForestClassifier()

#     # Define a list of k values to try
#     k_values = [5, 10, 15]
    
#     validation_accuracy = 0
    
#     best_k = 0
    
#     best_hyperparameters = {}

#     for k in k_values:
#         # Define the number of folds
#         kf = KFold(n_splits=k, shuffle=True, random_state=42)

#         # Use GridSearchCV for hyperparameter tuning with k-fold cross-validation
#         grid_search = GridSearchCV(random_forest, param_grid, cv=kf, scoring='accuracy', n_jobs=-1)
#         grid_search.fit(X_train_ohe, Y_train)
        
#         if grid_search.best_score_ > validation_accuracy:
#             validation_accuracy = grid_search.best_score_
#             best_k = k
#             best_hyperparameters = grid_search.best_params_
    
#     print(best_k)
#     print(best_hyperparameters)
#     print(validation_accuracy)
    
    # Best hyperparameters found during tuning
    best_hyperparameters = {'max_depth': None, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 100}
    
    random.seed(42)
    
    # Initialize the RandomForestClassifier with the best hyperparameters
    final_model = RandomForestClassifier(**best_hyperparameters, random_state=42)
    
    # Train the final model on the entire training dataset
    final_model.fit(X_train_ohe, Y_train)
    
    # Evaluate the model on the training data
    predictions = final_model.predict(X_train_ohe)
    print('classification report for Random Forest Classifier')
    print(f'{classification_report(Y_train, predictions)}')
    
    print(f'training accuracy for Random Forest Classifier {metrics.accuracy_score(Y_train, predictions)}')
    
    # Generate and display the confusion matrix
    cm = confusion_matrix(Y_train, predictions, labels=final_model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=final_model.classes_)
    disp.plot()
    plt.savefig('confusion_matrix_RFC.png')
    plt.show()

    # Make predictions on the test dataset
    Y_pred = final_model.predict(X_test_ohe)

    Y_pred = Y_pred.astype(int)

    # Create a submission file
    submission_data = pd.read_csv('test.csv')
    submission_data['Survived'] = pd.DataFrame(Y_pred, columns=['Survived'])['Survived'].astype(int)

    my_submission = pd.DataFrame({'PassengerId': submission_data.PassengerId, 'Survived': submission_data.Survived})
    my_submission.to_csv('random_forest.csv', index=False)
    
def svm_modelling(X_train_ohe, Y_train, X_test_ohe):
    """
    Perform Support Vector Machine (SVM) classification, including hyperparameter tuning, training the model, and making predictions.

    Parameters:
    - X_train_ohe: Training data features after one-hot encoding.
    - Y_train: Training data labels.
    - X_test_ohe: Test data features.

    Output:
    - Generates and saves a confusion matrix plot for the training data.
    - Outputs a classification report and training accuracy for the SVM Classifier.
    - Saves predictions for the test data in 'svm_submission.csv'.
    """
    # Assuming X_train_ohe and Y_train are your training data
    #Commented k-fold cross validation code
#     param_grid = {
#         'C': [0.1, 1, 10],
#         'kernel': ['linear', 'rbf', 'poly'],
#         'gamma': ['scale', 'auto']
#     }

#     svm_model = SVC()

#     random.seed(42)
    
#     # Define a list of k values to try
#     k_values = [5, 10, 15]

#     validation_accuracy = 0
#     best_k = 0
#     best_hyperparameters = {}

#     for k in k_values:
#         # Define the number of folds
#         kf = KFold(n_splits=k, shuffle=True, random_state=42)

#         # Use GridSearchCV for hyperparameter tuning with k-fold cross-validation
#         grid_search = GridSearchCV(svm_model, param_grid, cv=kf, scoring='accuracy', n_jobs=-1)
#         grid_search.fit(X_train_ohe, Y_train)

#         if grid_search.best_score_ > validation_accuracy:
#             validation_accuracy = grid_search.best_score_
#             best_k = k
#             best_hyperparameters = grid_search.best_params_
            
#     print(best_k)
#     print(best_hyperparameters)
#     print(validation_accuracy)

    # Best hyperparameters found during tuning
    best_hyperparameters = {'C': 1, 'gamma': 'scale', 'kernel': 'rbf'}
    
    random.seed(42)
    
    # Initialize the SVC with the best hyperparameters
    final_model = SVC(**best_hyperparameters, random_state=42)
    
    # Train the final model on the entire training dataset
    final_model.fit(X_train_ohe, Y_train)
    
    # Evaluate the model on the training data
    predictions = final_model.predict(X_train_ohe)
    print('classification report for SVC Classifier')
    print(f'{classification_report(Y_train, predictions)}')
    
    print(f'training accuracy for SVC Classifier {metrics.accuracy_score(Y_train, predictions)}')
    
    # Generate and display the confusion matrix
    cm = confusion_matrix(Y_train, predictions, labels=final_model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=final_model.classes_)
    disp.plot()
    plt.savefig('confusion_matrix_SVC.png')
    plt.show()

    # Make predictions on the test dataset
    Y_pred = final_model.predict(X_test_ohe)

    Y_pred = Y_pred.astype(int)

    # Create a submission file
    submission_data = pd.read_csv('test.csv')  # Replace 'test.csv' with the actual test dataset file
    submission_data['Survived'] = pd.DataFrame(Y_pred, columns=['Survived'])['Survived'].astype(int)

    my_submission = pd.DataFrame({'PassengerId': submission_data.PassengerId, 'Survived': submission_data.Survived})
    my_submission.to_csv('svm_submission.csv', index=False)
    
def mlp_modelling(X_train_ohe, Y_train, X_test_ohe):
    """
    Perform Multi-Layer Perceptron (MLP) classification, including hyperparameter tuning, training the model, and making predictions.

    Parameters:
    - X_train_ohe: Training data features after one-hot encoding.
    - Y_train: Training data labels.
    - X_test_ohe: Test data features.

    Output:
    - Generates and saves a confusion matrix plot for the training data.
    - Outputs a classification report and training accuracy for the MLP Classifier.
    - Saves predictions for the test data in 'mlp_submission.csv'.
    """
    # Assuming X_train_ohe and Y_train are your training data
    # Commmented k fold cross validation code
#     param_grid = {
#         'hidden_layer_sizes': [(100,), (50, 50), (100, 50, 25)],
#         'activation': ['relu', 'tanh'],
#         'alpha': [0.1, 0.01, 0.001],
#         'max_iter': [100, 200, 300],
#         'learning_rate': ['constant', 'invscaling', 'adaptive']
#     }

#     mlp_model = MLPClassifier()

#     random.seed(42)
    
#     # Define a list of k values to try
#     k_values = [5, 10, 15]
    
#     best_k = 0

#     validation_accuracy = 0
#     best_hyperparameters = {}

#     for k in k_values:
#         # Define the number of folds
#         kf = KFold(n_splits=k, shuffle=True, random_state=42)

#         # Use GridSearchCV for hyperparameter tuning with k-fold cross-validation
#         grid_search = GridSearchCV(mlp_model, param_grid, cv=kf, scoring='accuracy', n_jobs=-1)
#         grid_search.fit(X_train_ohe, Y_train)

#         if grid_search.best_score_ > validation_accuracy:
#             best_k = k
#             validation_accuracy = grid_search.best_score_
#             best_hyperparameters = grid_search.best_params_
    
#     print(best_k)
#     print(best_hyperparameters)
#     print(validation_accuracy)
    
    # Best hyperparameters found during tuning
    best_hyperparameters = {'activation': 'relu', 'alpha': 0.1, 'hidden_layer_sizes': (100,), 'learning_rate': 'adaptive', 'max_iter': 300}
    
    random.seed(42)
    
    # Initialize the MLPClassifier with the best hyperparameters
    final_model = MLPClassifier(**best_hyperparameters, random_state=42)
    
    # Train the final model on the entire training dataset
    final_model.fit(X_train_ohe, Y_train)
    
    # Evaluate the model on the training data
    predictions = final_model.predict(X_train_ohe)
    print("classification report for MLP Classifier")
    print(f'{classification_report(Y_train, predictions)}')
    
    print(f'training accuracy for MLP Classifier {metrics.accuracy_score(Y_train, predictions)}')
    
    # Generate and display the confusion matrix
    cm = confusion_matrix(Y_train, predictions, labels=final_model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=final_model.classes_)
    disp.plot()
    plt.savefig('confusion_matrix_MLP.png')
    plt.show()

    # Make predictions on the test dataset
    Y_pred = final_model.predict(X_test_ohe)

    Y_pred = Y_pred.astype(int)

    # Create a submission file
    submission_data = pd.read_csv('test.csv')  # Replace 'test.csv' with the actual test dataset file
    submission_data['Survived'] = pd.DataFrame(Y_pred, columns=['Survived'])['Survived'].astype(int)

    my_submission = pd.DataFrame({'PassengerId': submission_data.PassengerId, 'Survived': submission_data.Survived})
    my_submission.to_csv('mlp_submission.csv', index=False)


svm_modelling(X_train_ohe, Y_train, X_test_ohe)
mlp_modelling(X_train_ohe, Y_train, X_test_ohe)
random_forest_modelling(X_train_ohe, Y_train, X_test_ohe)
