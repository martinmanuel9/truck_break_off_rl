import numpy as np
import tensorflow as tf
import pandas as pd
import os
import random
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier  # Added import for RandomForestClassifier


class TruckBreakOffModel:
    def __init__(self):
        # Change the directory to 'dataset'
        print('Current directory:', os.getcwd())
        os.chdir('dataset')
        # Path to the CSV file
        csv_file = 'Truck_and_Bus_Through_Route.csv'

        # Read the CSV file into a dataframe
        df = pd.read_csv(csv_file)

        # define feature TRUCK_BREAK_OFF
        df['TRUCK_BREAK_OFF'] = 0
        # capture labels
        df['LABEL'] = 0
        df['LABEL'] = [random.randint(0, 1) for _ in range(len(df))]

        # Randomize 0s and 1s for the column TRUCK_BREAK_OFF
        df['TRUCK_BREAK_OFF'] = [random.randint(0, 1) for _ in range(len(df))]

        # Data preprocessing
        df['LAST_EDITED_DATE'] = pd.to_datetime(df['LAST_EDITED_DATE'])
        df['FROMDATE'] = pd.to_datetime(df['FROMDATE'])
        df['TODATE'] = pd.to_datetime(df['TODATE'])
        # Convert datetime to Unix timestamp
        df['LAST_EDITED_DATE'] = df['LAST_EDITED_DATE'].astype(int)
        df['FROMDATE'] = df['FROMDATE'].astype(int)
        df['TODATE'] = df['TODATE'].astype(int)
        df['FROMMEASURE'] = df['FROMMEASURE'].astype(int)
        df['TOMEASURE'] = df['TOMEASURE'].astype(int)
        df['ROUTEID'] = df['ROUTEID'].astype('category').cat.codes
        ## normalization
        scaler = MinMaxScaler()
        df[['ROUTEID', 'LAST_EDITED_DATE', 'TRUCK_BREAK_OFF', 'FROMDATE', 'TODATE', 'FROMMEASURE', 'TOMEASURE']] = scaler.fit_transform(
            df[['ROUTEID', 'LAST_EDITED_DATE', 'TRUCK_BREAK_OFF', 'FROMDATE', 'TODATE', 'FROMMEASURE', 'TOMEASURE']])

        # Data preprocessing complete
        print('Dataset:\n', df.head(5))
        # create train test split
        self.train, self.test = train_test_split(df, test_size=0.2, random_state=200)

        print('Train set:\n', self.train.head())
        print('Test set:\n', self.test.head())

        self.features = ['ROUTEID', 'LAST_EDITED_DATE', 'FROMMEASURE', 'TOMEASURE']
        self.target = 'TRUCK_BREAK_OFF'



    def ml_model(self):  # Fixed indentation
        num_episodes = 1000
        epsilon = 0.1

        # Create a random forest classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42)

        # Generate random data
        X = np.random.rand(100, 4)  # Example data, replace with actual data
        y = np.random.randint(2, size=100)  # Example labels, replace with actual labels

        # Train the classifier
        clf.fit(X, y)
        return clf  # Return the trained model

    def evaluate_model(self, model):
        # Evaluate the model on the test set
        X_test = self.test[self.features].values  # Use only the selected features for testing and convert to numpy array
        y_true = self.test['LABEL']
        y_pred = model.predict(X_test)  # Use predict method of RandomForestClassifier
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, zero_division=1)  # Set zero_division parameter
        confusion = confusion_matrix(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=1)  # Set zero_division parameter
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        print("Accuracy:", accuracy)
        print("Classification Report:\n", report)
        print("Confusion Matrix:\n", confusion)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)


test = TruckBreakOffModel()
trained_model = test.ml_model()  # Corrected method call from 'reinforcement_model()' to 'ml_model()'
test.evaluate_model(trained_model)
