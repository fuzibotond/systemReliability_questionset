import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder


# Function to generate continuous data within the specified range
def generate_continuous_data(length, min_value, max_value, percentage_range):
    data_range = (max_value - min_value) * percentage_range / 100
    data = np.random.uniform(min_value, min_value + data_range, length)
    return data

# Function to visualize the dataset graphically in continuous form
def visualize_continuous_data(continuous_data):
    plt.figure(figsize=(10, 6))
    for i, sequence in enumerate(continuous_data):
        plt.plot(sequence, label=f'Sequence {i+1}')
    plt.xlabel('Time')
    plt.ylabel('Temperature (°C)')
    plt.title('Continuous Dataset')
    plt.legend()
    plt.show()

# Function to discretize the continuous data into three categories
def discretize_data(continuous_data):
    discrete_data = []
    average_temp = (max_temp + min_temp) / 2
    for sequence in continuous_data:
        discrete_sequence = np.where(sequence <= average_temp, 'Cold', 'Hot')
        discrete_sequence = np.where((sequence >= 9) & (sequence <= 21), 'Average', discrete_sequence)
        discrete_data.append(discrete_sequence)
    return discrete_data

# Function to visualize the dataset graphically in discrete form
def visualize_discrete_data(discrete_data):
    plt.figure(figsize=(10, 6))
    for i, sequence in enumerate(discrete_data):
        plt.plot(sequence, label=f'Sequence {i+1}')
    plt.xlabel('Time')
    plt.ylabel('Temperature Category')
    plt.title('Discrete Dataset')
    plt.legend()
    plt.show()

# Function to train an HMM using the discrete dataset
# Function to train an HMM using the discrete dataset
def train_hmm(discrete_data):
    label_encoder = LabelEncoder()
    discrete_data_encoded = [label_encoder.fit_transform(seq) for seq in discrete_data]

    model = hmm.MultinomialHMM(n_components=3)
    X_lengths = [len(seq) for seq in discrete_data_encoded]
    X_flat = np.concatenate(discrete_data_encoded)
    model.fit(X_flat.reshape(-1, 1), X_lengths)
    return model



# Function to print the model parameters
def print_model_parameters(trained_model):
    print("Initial state distribution (π):")
    print(trained_model.startprob_)
    print("\nTransition matrix (A):")
    print(trained_model.transmat_)
    print("\nEmission matrix (B):")
    print(trained_model.emissionprob_)

# Generate continuous data sequences
num_sequences = 5
sequence_length = 150
min_temp = -12
max_temp = 37
temperature_range_percentage = 60
continuous_data = []
for _ in range(num_sequences):
    data = generate_continuous_data(sequence_length, min_temp, max_temp, temperature_range_percentage)
    continuous_data.append(data)

# Visualize the dataset graphically in continuous form
visualize_continuous_data(continuous_data)

# Discretize the continuous data into three categories
discrete_data = discretize_data(continuous_data)

# Visualize the dataset graphically in discrete form
visualize_discrete_data(discrete_data)

# Train an HMM using the discrete dataset
trained_model = train_hmm(discrete_data)

# Print model parameters
print_model_parameters(trained_model)
