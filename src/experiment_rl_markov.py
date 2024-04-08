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

        self.features = ['ROUTEID', 'LAST_EDITED_DATE', 'FROMDATE', 'TODATE', 'FROMMEASURE', 'TOMEASURE']
        self.target = 'TRUCK_BREAK_OFF'


        # Define the transition matrix (Markov chain)
        self.transition_matrix = np.array([[0.9, 0.1],
                                           [0.3, 0.7]])

        # Define the reward matrix
        self.reward_matrix = np.array([[10, -1],
                                        [-1, 10]])

    def reinforcement_model(self):
        # Define hyperparameters
        num_episodes = 1000
        learning_rate = 0.1
        discount_factor = 0.95
        epsilon = 0.1

        # Define the Q-network
        num_states = self.transition_matrix.shape[0]
        num_actions = self.transition_matrix.shape[1]
        num_features = 7  # Number of features in your input data
        W = tf.Variable(tf.random.uniform([num_states, num_actions], 0, 0.01))
        W = tf.transpose(W)

        # Define loss and optimizer
        optimizer = tf.optimizers.SGD(learning_rate=learning_rate)

        # Initialize TensorFlow session
        for episode in range(num_episodes):
            state = np.random.randint(0, num_states)  # Start at a random state
            one_hot_state = 0.0 
            while True:
                # Choose action (epsilon-greedy)
                if np.random.rand() < epsilon:
                    action = np.random.randint(0, num_actions)
                else:
                    # one_hot_state = tf.reshape(tf.one_hot(state, num_states), [1, -1])
                    one_hot_state = tf.reshape(tf.one_hot(state, num_states), [1, num_states])
                    print(one_hot_state.shape)
                    print(W.shape)
                    
                    # Perform matrix multiplication
                    action = tf.argmax(tf.matmul(one_hot_state, W), 1).numpy()[0]

                # Perform action and observe next state and reward
                next_state = np.random.choice(range(num_states), p=self.transition_matrix[state])
                hot_next_state = tf.reshape(tf.one_hot(next_state, num_states), [1, -1])
                reward = self.reward_matrix[state, action]
                # Compute Q-value of next state
                Q_next = tf.matmul(hot_next_state, W)
                # Update Q-value of current state
                max_Q_next = tf.reduce_max(Q_next)
                target_Q_values = tf.matmul(hot_next_state, W)

                # Update Q-value of current state
                target_Q_values_updated = tf.identity(target_Q_values)  # Create a copy
                target_Q_values_updated = tf.tensor_scatter_nd_update(target_Q_values_updated, [[0, action]],
                                                                      [reward + discount_factor * max_Q_next])

                # Train Q-network
                with tf.GradientTape() as tape:
                    Q_values = tf.matmul(one_hot_state, W)
                    loss = tf.reduce_sum(tf.square(target_Q_values_updated - Q_values))

                gradients = tape.gradient(loss, [W])
                optimizer.apply_gradients(zip(gradients, [W]))
                state = next_state
                if state == 0:  # Reached terminal state
                    break
        # Save the learned model
        tf.saved_model.save(W, '../model/truck_break_off_model')
        # tf.saved_model.save(W, os.path.join('/opt/ml/model', 'truck_break_off_model'))

        # Print the learned Q-values
        print("Learned Q-values:")
        print(W.numpy())
        return W

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
trained_model = test.reinforcement_model()
test.evaluate_model(trained_model)
