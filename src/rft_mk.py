import numpy as np
import tensorflow as tf
import pandas as pd
import os
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier

class TruckBreakOffModel:
    def __init__(self):
        # Change the directory to 'dataset'
        print('Current directory:', os.getcwd())
        os.chdir('dataset')
        # Path to the CSV file
        csv_file = 'Truck_and_Bus_Through_Route.csv'

        # Read the CSV file into a dataframe
        df = pd.read_csv(csv_file)

        # Define feature TRUCK_BREAK_OFF
        df['TRUCK_BREAK_OFF'] = [random.randint(0, 1) for _ in range(len(df))]

        # Capture labels
        df['LABEL'] = [random.randint(0, 1) for _ in range(len(df))]

        # Data preprocessing
        df['LAST_EDITED_DATE'] = pd.to_datetime(df['LAST_EDITED_DATE']).astype(int)
        df['FROMDATE'] = pd.to_datetime(df['FROMDATE']).astype(int)
        df['TODATE'] = pd.to_datetime(df['TODATE']).astype(int)
        df['FROMMEASURE'] = df['FROMMEASURE'].astype(int)
        df['TOMEASURE'] = df['TOMEASURE'].astype(int)
        df['ROUTEID'] = df['ROUTEID'].astype('category').cat.codes

        # Normalization
        scaler = MinMaxScaler()
        df[['ROUTEID', 'LAST_EDITED_DATE', 'TRUCK_BREAK_OFF', 'FROMDATE', 'TODATE', 'FROMMEASURE', 'TOMEASURE']] = scaler.fit_transform(
            df[['ROUTEID', 'LAST_EDITED_DATE', 'TRUCK_BREAK_OFF', 'FROMDATE', 'TODATE', 'FROMMEASURE', 'TOMEASURE']])

        # Display dataset info
        print('Dataset:\n', df.head(5))
        # Create train test split
        self.train, self.test = train_test_split(df, test_size=0.2, random_state=200)

        print('Train set:\n', self.train.head())
        print('Test set:\n', self.test.head())

        # Features and target variable
        self.features = ['ROUTEID', 'LAST_EDITED_DATE', 'FROMMEASURE', 'TOMEASURE']
        self.target = 'TRUCK_BREAK_OFF'

        # Define the transition matrix (Markov chain)
        self.transition_matrix = np.array([[0.9, 0.1],
                                            [0.3, 0.7]])

        # Define the reward matrix
        self.reward_matrix = np.array([[10, -1],
                                        [-1, 10]])

    def ml_model(self):
        # Create a random forest classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42)

        # Generate random data (dummy data for illustration, replace with actual data)
        X = np.random.rand(100, len(self.features))
        y = np.random.randint(2, size=100)

        # Train the classifier
        clf.fit(X, y)
        return clf

    def evaluate_model(self, model):
        # Evaluate the model on the test set
        X_test = self.test[self.features].values
        y_true = self.test['LABEL']
        y_pred = model.predict(X_test)

        # Integrate Markov chain logic for prediction correction
        corrected_predictions = []
        for index, row in self.test.iterrows():
            state = int(row['TRUCK_BREAK_OFF'])
            action = self.markov_chain(state)
            corrected_predictions.append(action)

        corrected_accuracy = accuracy_score(y_true, corrected_predictions)
        print("Corrected Accuracy (incorporating Markov Chain):", corrected_accuracy)

        # Calculate other metrics
        corrected_report = classification_report(y_true, corrected_predictions, zero_division=1)
        corrected_confusion = confusion_matrix(y_true, corrected_predictions)
        corrected_precision = precision_score(y_true, corrected_predictions, zero_division=1)
        corrected_recall = recall_score(y_true, corrected_predictions)
        corrected_f1 = f1_score(y_true, corrected_predictions)

        print("Classification Report (incorporating Markov Chain):\n", corrected_report)
        print("Confusion Matrix (incorporating Markov Chain):\n", corrected_confusion)
        print("Precision (incorporating Markov Chain):", corrected_precision)
        print("Recall (incorporating Markov Chain):", corrected_recall)
        print("F1 Score (incorporating Markov Chain):", corrected_f1)

    def markov_chain(self, state):
        next_state = np.random.choice([0, 1], p=self.transition_matrix[state])
        return next_state

test = TruckBreakOffModel()
trained_model = test.ml_model()
test.evaluate_model(trained_model)
