
import numpy as np
import pandas as pd
import os
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import joblib
import argparse

class TruckBreakOffModel:

    def model_fn(self, model_dir):
        clf = joblib.load(os.path.join(model_dir, "model.joblib"))
        return clf


    def ml_model(self):  
        n_estimators = 100
        random_state = 0  

        # Create a random forest classifier
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

        # Generate random data
        X = np.random.rand(100, 4)  # Example data, replace with actual data
        y = np.random.randint(2, size=100)  # Example labels, replace with actual labels

        # Train the classifier
        clf.fit(X, y)

        # Evaluate the model:
        y_pred = clf.predict(X)
        accuracy = accuracy_score(y, y_pred)
        print("Accuracy:", accuracy)
        # classification report
        report = classification_report(y, y_pred)
        print("Classification Report:\n", report)
        # confusion matrix
        confusion = confusion_matrix(y, y_pred)
        print("Confusion Matrix:\n", confusion)
        # precision
        precision = precision_score(y, y_pred)
        print("Precision:", precision)
        # recall
        recall = recall_score(y, y_pred)
        print("Recall:", recall)
        # f1 score
        f1 = f1_score(y, y_pred) 
        print("F1 Score:", f1)

        
        # Save the trained model
        joblib.dump(clf, "model.joblib")
        return clf  

    

        

if __name__ == "__main__":
    print("[INFO] Extracting arguments...")
    truck_break_off_mdl = TruckBreakOffModel()

    parser = argparse.ArgumentParser()

    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=0)

    # Data, model, and output directories
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAINING"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TESTING"))
    parser.add_argument("--train-file", type=str, default="train-V1.csv")
    parser.add_argument("--test-file", type=str, default="test-V1.csv")

    args, _ = parser.parse_known_args()

    print("[INFO] Reading data...")
    print()
    train_df = pd.read_csv(os.path.join(args.train, args.train_file))
    test_df = pd.read_csv(os.path.join(args.test, args.test_file))

    print("Train Dataset:\n", train_df.head())
    print()
    print("Test Dataset:\n", test_df.head())
    print()

    print("[INFO] Building Training & Testing Datasets...")
    print()
    features = ['ROUTEID', 'LAST_EDITED_DATE','FROMDATE', 'TODATE', 'FROMMEASURE', 'TOMEASURE', 'TRUCK_BREAK_OFF']
    label = 'LABEL'

    print("[INFO] Training Model...")
    print()

    model = truck_break_off_mdl.ml_model()  
    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)  
    print("Model saved at: {}".format(model_path))
    print()
