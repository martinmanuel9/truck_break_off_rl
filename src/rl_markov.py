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
        # Convert datetime to Unix timestamp
        df['LAST_EDITED_DATE'] = df['LAST_EDITED_DATE'].astype(int)
        df['ROUTEID'] = df['ROUTEID'].astype('category').cat.codes

        ## normalization
        scaler = MinMaxScaler()
        df[['ROUTEID', 'LAST_EDITED_DATE', 'TRUCK_BREAK_OFF']] = scaler.fit_transform(
            df[['ROUTEID', 'LAST_EDITED_DATE', 'TRUCK_BREAK_OFF']])

        # Data preprocessing complete
        print('Dataset:\n', df.head(5))
        # create train test split
        self.train, self.test = train_test_split(df, test_size=0.2, random_state=200)

        print('Train set:\n', self.train.head())
        print('Test set:\n', self.test.head())



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
        num_features = 4  # Number of features in your input data
        W = tf.Variable(tf.random.uniform([num_states, num_actions], 0, 0.01))

        # Define loss and optimizer
        optimizer = tf.optimizers.SGD(learning_rate=learning_rate)

        # Initialize TensorFlow session
        for episode in range(num_episodes):
            state = np.random.randint(0, num_states)  # Start at a random state
            while True:
                # Choose action (epsilon-greedy)
                if np.random.rand() < epsilon:
                    action = np.random.randint(0, num_actions)
                else:
                    one_hot_state = tf.reshape(tf.one_hot(state, num_states), [1, -1])
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

        # Print the learned Q-values
        print("Learned Q-values:")
        print(W.numpy())
        return W

    def evaluate_model(self, model):
        # Evaluate the model on the test set
        num_states = self.transition_matrix.shape[0]
        num_actions = self.transition_matrix.shape[1]
        num_features = 4  # Number of features in your input data
        correct_predictions = 0
        for index, row in self.test.iterrows():
            state = int(row['ROUTEID'])  # Convert state to integer
            one_hot_state = tf.reshape(tf.one_hot(state, num_states), [1, -1])
            action = tf.argmax(tf.matmul(one_hot_state, model), 1).numpy()[0]
            # Assuming action 0 corresponds to no truck break off, action 1 corresponds to truck break off
            predicted_break_off = action
            true_break_off = row['LABEL']
            if predicted_break_off == true_break_off:
                correct_predictions += 1

        manual_calc_accuracy = correct_predictions / len(self.test)
        print("Manual calculation accuracy:", manual_calc_accuracy)

        # Evaluate the model using sklearn metrics
        y_true = self.test['LABEL']
        y_pred = []
        for index, row in self.test.iterrows():
            state = int(row['ROUTEID'])  # Convert state to integer
            one_hot_state = tf.reshape(tf.one_hot(state, num_states), [1, -1])
            action = tf.argmax(tf.matmul(one_hot_state, model), 1).numpy()[0]
            # Assuming action 0 corresponds to no truck break off, action 1 corresponds to truck break off
            predicted_break_off = action
            y_pred.append(predicted_break_off)
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred)
        confusion = confusion_matrix(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        print("Accuracy:", accuracy)
        print("Classification Report:\n", report)
        print("Confusion Matrix:\n", confusion)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)


test = TruckBreakOffModel()
trained_model = test.reinforcement_model()
test.evaluate_model(trained_model)
