
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import random
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import joblib
import pathlib
from io import StringIO
import argparse
import joblib

class TruckBreakOffModel:
    # saves model within s3 bucket
    def model_fn(self, model_dir):
        clf = joblib.load(os.path.join(model_dir, "model.joblib"))
        return clf


    def reinforcement_model(self):
        # Define markov chain
        # Define the transition matrix (Markov chain)
        self.transition_matrix = np.array([[0.9, 0.1],
                                    [0.3, 0.7]])

        # Define the reward matrix
        self.reward_matrix = np.array([[10, -1],
                                [-1, 10]])

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
        W = tf.transpose(W)

        # Define loss and optimizer
        optimizer = tf.optimizers.SGD(learning_rate=learning_rate)

        # Initialize TensorFlow session
        for episode in range(num_episodes):
            state = np.random.randint(0, num_states)  # Start at a random state
            one_hot_state = 0.0  # Initialize one_hot_state outside the if-else block
            while True:
                # Choose action (epsilon-greedy)
                if np.random.rand() < epsilon:
                    action = np.random.randint(0, num_actions)
                else:
                    one_hot_state = tf.reshape(tf.one_hot(state, num_states), [1, -1])
                    # one_hot_state = tf.reshape(tf.one_hot(state, num_states), [1, 1, num_states])

                    action = tf.argmax(tf.matmul(one_hot_state, W), 1).numpy()[0]
                # Perform action and observe next state and reward
                next_state = np.random.choice(range(num_states), p= self.transition_matrix[state])
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
       
        # Print the learned Q-values
        print("Learned Q-values:")
        print(W.numpy())




if __name__ == "__main__":
    print("[INFO] Extracting arguments...")
    parser = argparse.ArgumentParser()
    truck_break_off_mdl = TruckBreakOffModel()
    truck_break_off_mdl.reinforcement_model()
    transition_matrix = truck_break_off_mdl.transition_matrix
    reward_matrix = truck_break_off_mdl.reward_matrix

    # Hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--discount_factor", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--num_states", type=int, default=transition_matrix.shape[0])
    parser.add_argument("--num_actions", type=int, default=transition_matrix.shape[1])
    parser.add_argument("--num_features", type=int, default=7)


    # Data, model, and output directories
    # sets the SageMaker environment within SageMaker
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAINING"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TESTING"))

    # test/train files
    # parser.add_argument("--train_file", type=str, default="s3://martymdlregistry/sagemaker/truck-break-off-rl_markov/datasets/train-V1.csv")
    # parser.add_argument("--test_file", type=str, default="s3://martymdlregistry/sagemaker/truck-break-off-rl_markov/datasets/test-V1.csv")

    parser.add_argument("--train_file", type=str, default="train-V1.csv")
    parser.add_argument("--test_file", type=str, default="test-V1.csv")


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
    X_train = train_df[features]
    y_train = train_df[label]
    X_test = test_df[features]
    y_test = test_df[label]

    print("[INFO] Training Model...")
    print()

    # send to S3 bucket. SageMaker will take training data from the S3 bucket
    sk_prefix = "sagemaker/truck-break-off-rl_markov/datasets" # sagemaker environment
    model_dir = args.model_dir
    truck_break_off_mdl.reinforcement_model()
    truck_break_off_mdl.model_fn(model_dir)
    

    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(truck_break_off_mdl.reinforcement_model, model_path)
    print("Model saved at: {}".format(model_path))
    print()

    print("[INFO] Model Training Complete...")
