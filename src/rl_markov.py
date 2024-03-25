import numpy as np
import tensorflow as tf
import pandas as pd
import os
import random
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder


class TruckBreakOffModel:
    def __init__(self):
        # Change the directory to 'dataset'
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
        df[['ROUTEID','LAST_EDITED_DATE','TRUCK_BREAK_OFF']] = scaler.fit_transform(df[['ROUTEID','LAST_EDITED_DATE','TRUCK_BREAK_OFF']])

        # Data preprocessing complete
        # print('Dataset:\n', df.head(5))
        # create train test split
        self.train = df.sample(frac=0.8, random_state=200)
        self.test = df.drop(self.train.index)

        # print('Train set:\n', self.train.head())
        # print('Test set:\n', self.test.head())


        
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
                    # print("Shape of W:", W.shape)
                    # print(tf.one_hot(state, num_states))
                    one_hot_state = tf.reshape(tf.one_hot(state, num_states), [1, -1])
                    # print(tf.one_hot(state, num_states))
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
                target_Q_values_updated = tf.tensor_scatter_nd_update(target_Q_values_updated, [[0, action]], [reward + discount_factor * max_Q_next])

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
        # Print the learned Q-values
        print("Learned Q-values:")
        print(W.numpy())



test = TruckBreakOffModel()
test.reinforcement_model()
