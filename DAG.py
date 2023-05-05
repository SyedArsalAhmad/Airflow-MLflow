#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import mlflow
from mlflow.tracking import MlflowClient
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib

# Define default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 5, 5),
    'retries': 1
}

# Define the DAG
dag = DAG('titanic_pipeline', default_args=default_args)

# Define the data preprocessing step as a PythonOperator
def preprocess_data():
    # Read in the dataset
    train_df = pd.read_csv('complete.csv')

    # Drop unnecessary columns
    train_df = train_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

    # Replace missing values in 'Age' with the median age
    train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
    train_df['Fare'].fillna(train_df['Fare'].median(), inplace=True)

    # Replace missing values in 'Embarked' with the most common value
    train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)

    # Convert categorical features to numerical
    train_df['Sex'] = train_df['Sex'].map({'female': 1, 'male': 0})
    train_df['Embarked'] = train_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    # Create a new feature 'FamilySize' by combining 'SibSp' and 'Parch'
    train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1

    # Create a new feature 'IsAlone' indicating whether the passenger is alone or not
    train_df['IsAlone'] = 0
    train_df.loc[train_df['FamilySize'] == 1, 'IsAlone'] = 1

    # Drop 'SibSp' and 'Parch' columns
    train_df = train_df.drop(['SibSp', 'Parch'], axis=1)

    # Save the preprocessed data to a new file
    train_df.to_csv('preprocessed_data.csv', index=False)

preprocess_data_task = PythonOperator(task_id='preprocess_data', 
                                      python_callable=preprocess_data, 
                                      dag=dag)

# Define the model training step as a PythonOperator
def train_model():
    # Read in the preprocessed data
    train_df = pd.read_csv('preprocessed_data.csv')

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(train_df.drop('Survived', axis=1),
                                                        train_df['Survived'], test_size=0.2, random_state=42)

    # Create a logistic regression model and fit the training data
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Save the trained model to a new file
    joblib.dump(model, 'model.joblib')
    
    # Log model parameters and metrics to MLflow
    with mlflow.start_run() as run:
        mlflow.log_params({'random_state': 42, 'test_size': 0.2})
        mlflow.log_metric('accuracy', accuracy_score(y_test, model.predict(X_test)))
        mlflow.log_artifact('model.joblib')
        mlflow.log_artifact('preprocessed_data.csv')

train_model_task = PythonOperator(task_id='train_model', 
                                   python_callable=train_model, 
                                   dag=dag)

# Define the model evaluation step as a PythonOperator
def evaluate_model():
    # Read in the preprocessed data and the trained model
    train_df = pd.read_csv('preprocessed_data.csv')
    model = joblib.load('model.joblib')
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(train_df.drop('Survived', axis=1),
                                                        train_df['Survived'], test_size=0.2, random_state=42)

    # Make predictions on the testing set
    y_pred = model.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)

    # Print the accuracy of the model
    print('Model accuracy: {:.2f}%'.format(accuracy * 100))
    
    mlflow.log_metric('accuracy', accuracy)
    
evaluate_model_task = PythonOperator(task_id='evaluate_model',
python_callable=evaluate_model,
dag=dag)

deploy_model_task = BashOperator(task_id='deploy_model',
bash_command='echo "Model deployed"',
dag=dag)

preprocess_data_task >> train_model_task >> evaluate_model_task >> deploy_model_task ## Complete pipeline

