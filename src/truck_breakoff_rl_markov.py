
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

# Disable eager execution
tf.compat.v1.disable_eager_execution()

class TruckBreakOffModel:
    # saves model within s3 bucket
    def model_fn(self, model_dir):
        clf = joblib.load(os.path.join(model_dir, "model.joblib"))
        return clf


    def reinforcement_model(self, args):
        # Define markov chain
        transition_matrix = np.array([[0.9, 0.1],
                                      [0.3, 0.7]])

        # Define the reward matrix
        reward_matrix = np.array([[10, -1],
                                  [-1, 10]])

        # Define hyperparameters
        num_episodes = args.num_episodes
        learning_rate = args.learning_rate
        discount_factor = args.discount_factor
        epsilon = args.epsilon

        # Define the Q-network
        num_states = transition_matrix.shape[0]
        num_actions = transition_matrix.shape[1]
        W = tf.Variable(tf.random.uniform([num_states, num_actions], 0, 0.01))
        W = tf.transpose(W)

        # Define placeholders for state, action, and target Q-value
        state_ph = tf.compat.v1.placeholder(tf.int32, shape=[])
        action_ph = tf.compat.v1.placeholder(tf.int32, shape=[])
        target_q_value_ph = tf.compat.v1.placeholder(tf.float32, shape=[])

        # Compute Q-value of current state
        one_hot_state = tf.one_hot(state_ph, num_states)
        one_hot_state = tf.reshape(one_hot_state, [1, -1])  # Reshape to match the shape of W
        Q_values = tf.matmul(one_hot_state, tf.transpose(W))

        # Define loss
        updated_Q_values = tf.tensor_scatter_nd_update(Q_values, [[0, action_ph]], [target_q_value_ph])
        loss = tf.reduce_sum(tf.square(updated_Q_values - Q_values))

        # Define optimizer
        optimizer = tf.optimizers.SGD(learning_rate=learning_rate)

        # Define training operation
        train_op = optimizer.minimize(loss)

        # Start TensorFlow session
        with tf.compat.v1.Session() as sess:
            # Initialize variables
            sess.run(tf.compat.v1.global_variables_initializer())

            # Training loop
            for episode in range(num_episodes):
                state = np.random.randint(0, num_states)  # Start at a random state
                while True:
                    # Choose action (epsilon-greedy)
                    if np.random.rand() < epsilon:
                        action = np.random.randint(0, num_actions)
                    else:
                        action = sess.run(tf.argmax(Q_values, 1), feed_dict={state_ph: state})

                    # Perform action and observe next state
                    next_state = np.random.choice(range(num_states), p=transition_matrix[state])

                    # Compute reward
                    reward = reward_matrix[state, action]

                    # Compute target Q-value
                    max_Q_next = np.max(sess.run(Q_values, feed_dict={state_ph: next_state}))
                    target_Q_value = reward + discount_factor * max_Q_next

                    # Update Q-value
                    _ = sess.run(train_op, feed_dict={state_ph: state, action_ph: action, target_q_value_ph: target_Q_value})

                    state = next_state
                    if state == 0:  # Reached terminal state
                        break

            # Get learned Q-values
            learned_Q_values = sess.run(Q_values)
       

        # Print the learned Q-values
        print("Learned Q-values:")
        print(learned_Q_values)
        # After training is complete, assign the learned values to instance attributes
        self.learned_Q_values = learned_Q_values
        self.transition_matrix = transition_matrix
        self.reward_matrix = reward_matrix

        return learned_Q_values, transition_matrix, reward_matrix

    def evaluate_model(self, learned_Q_values, transition_matrix, reward_matrix):
        num_states = transition_matrix.shape[0]
        num_actions = transition_matrix.shape[1]
        
        # Initialize the cumulative return
        total_return = 0

        # Run episodes to compute the return
        num_episodes = 1000  # You can adjust this number
        for _ in range(num_episodes):
            state = np.random.randint(0, num_states)
            episode_return = 0
            while True:
                action = np.argmax(learned_Q_values[state])
                next_state = np.random.choice(range(num_states), p=transition_matrix[state])
                reward = reward_matrix[state, action]
                episode_return += reward
                state = next_state
                if state == 0:  # Reached terminal state
                    break
            total_return += episode_return

        average_return = total_return / num_episodes
        return average_return


if __name__ == "__main__":
    print("[INFO] Extracting arguments...")
    print()
    truck_break_off_mdl = TruckBreakOffModel()

    parser = argparse.ArgumentParser()


    # Hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--discount_factor", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--num_states", type=int, default=2)
    parser.add_argument("--num_actions", type=int, default=2)
    parser.add_argument("--num_features", type=int, default=7)
 
    # Data, model, and output directories
    # sets the SageMaker environment within SageMaker
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAINING"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TESTING"))

    # test/train files
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
    sk_prefix = "sagemaker/truck-break-off/datasets" # sagemaker environment
    model_dir = args.model_dir

    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(truck_break_off_mdl.reinforcement_model, model_path)
    print("Model saved at: {}".format(model_path))
    print()

    print("[INFO] Model Training Complete...")

 
