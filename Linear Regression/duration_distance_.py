import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# Load the rides_data2.csv file
data = pd.read_csv('rides_data2.csv')

# Print the first few rows of the data to understand its structure
print(data.head())

# Select features (e.g., 'Duration' and 'Distance') and target ('Fare')
# Ensure these columns exist in your data, otherwise adjust accordingly
x_data = data[['Duration_min', 'Distance_km']].values  # Features
y_data = data['Fare'].values  # Target (Fare)

# Normalize the data (optional but often recommended for gradient descent)
x_data = (x_data - np.mean(x_data, axis=0)) / np.std(x_data, axis=0)

# Plot the data (Optional: You can plot the data for one feature at a time, but here we use 2 features)
plt.scatter(x_data[:, 0], y_data, color='red', label='Duration vs Fare')
plt.scatter(x_data[:, 1], y_data, color='blue', label='Distance vs Fare')
plt.xlabel('Duration / Distance')
plt.ylabel('Fare')
plt.title('Input Data')
plt.legend()
plt.show()

# Generate weights and biases for two features (2 weights for 'Duration' and 'Distance')
W = tf.Variable(tf.random_uniform([2], -1.0, 1.0))  # Weight for each feature
b = tf.Variable(tf.zeros([1]))  # Bias term

# Define the model equation: y = W1 * Duration + W2 * Distance + b
y_pred = tf.reduce_sum(W * x_data, axis=1) + b  # Predicted Fare values

# Define the loss function: Mean squared error
loss = tf.reduce_mean(tf.square(y_pred - y_data))

# Define the gradient descent optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# Initialize all the variables
init = tf.global_variables_initializer()

# Start the TensorFlow session
with tf.Session() as sess:
    sess.run(init)
    
    # Train the model for a specified number of iterations
    num_iterations = 100
    for step in range(num_iterations):
        # Run the training step
        sess.run(train)

        # Print the progress every 10 steps
        if (step + 1) % 10 == 0:
            print(f'\nIteration {step+1}')
            print(f'W = {sess.run(W)}')
            print(f'b = {sess.run(b)}')
            print(f'Loss = {sess.run(loss)}')

            # Plot the model prediction vs actual data for current iteration
            plt.scatter(x_data[:, 0], y_data, color='red', label='Duration vs Fare')
            plt.scatter(x_data[:, 1], y_data, color='blue', label='Distance vs Fare')
            plt.plot(x_data[:, 0], sess.run(W)[0] * x_data[:, 0] + sess.run(b), label="Predicted Line (Duration)", color='green')
            plt.plot(x_data[:, 1], sess.run(W)[1] * x_data[:, 1] + sess.run(b), label="Predicted Line (Distance)", color='orange')
            plt.xlabel('Duration / Distance')
            plt.ylabel('Fare')
            plt.title(f'Iteration {step+1} of {num_iterations}')
            plt.legend()
            plt.show()

    # Final weights and bias
    print(f'\nFinal Model: W = {sess.run(W)}')
    print(f'Final Bias: b = {sess.run(b)}')
    print(f'Final Loss: {sess.run(loss)}')
