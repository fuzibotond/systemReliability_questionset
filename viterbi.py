import numpy as np
import matplotlib.pyplot as plt

# Trained parameters
pi = np.array([0.0663392, 0.9179163, 0.0157445])
A = np.array([[5.30558173e-01, 2.45157417e-01, 2.24284410e-01],
              [7.01106139e-01, 2.68719225e-01, 3.01746352e-02],
              [9.89195013e-01, 1.11769270e-04, 1.06932175e-02]])
B = np.array([[1.],
              [1.],
              [1.]])

# Observations
observations = ["Average", "Cold", "Average", "Hot"]
num_observations = len(observations)

# Initialize Viterbi probabilities and backpointers
viterbi = np.zeros((num_observations, 3))
backpointers = np.zeros((num_observations, 3), dtype=int)

# Step 1: Initialize probabilities for the first observation
obs_idx = 0
for state in range(3):
    viterbi[obs_idx][state] = pi[state] * B[state][0]

# Step 2-3: Calculate probabilities for subsequent observations
for obs_idx in range(1, num_observations):
    obs = observations[obs_idx]
    for current_state in range(3):
        max_prob = -1
        max_state = -1
        for prev_state in range(3):
            prob = viterbi[obs_idx - 1][prev_state] * A[prev_state][current_state] * B[current_state][0]
            if prob > max_prob:
                max_prob = prob
                max_state = prev_state
        viterbi[obs_idx][current_state] = max_prob
        backpointers[obs_idx][current_state] = max_state

# Step 4: Trace back to find the most likely sequence of hidden states
best_sequence = []
max_last_prob = -1
best_last_state = -1
for state in range(3):
    if viterbi[num_observations - 1][state] > max_last_prob:
        max_last_prob = viterbi[num_observations - 1][state]
        best_last_state = state
best_sequence.append(best_last_state)
for obs_idx in range(num_observations - 1, 0, -1):
    best_state = backpointers[obs_idx][best_last_state]
    best_sequence.insert(0, best_state)
    best_last_state = best_state

# Plot Viterbi diagram
plt.figure(figsize=(10, 6))
plt.title('Viterbi Diagram')
plt.xlabel('Observations')
plt.ylabel('State')
for state in range(3):
    plt.plot(range(num_observations), [state]*num_observations, linestyle='-', marker='o', label=f'State {state+1}')
plt.yticks(range(3), ['State 1', 'State 2', 'State 3'])
plt.xticks(range(num_observations), observations)
for obs_idx in range(1, num_observations):
    for state in range(3):
        prev_state = backpointers[obs_idx][state]
        plt.plot([obs_idx - 1, obs_idx], [prev_state, state], linestyle='-', color='gray')
plt.scatter(range(num_observations), best_sequence, color='red', label='Best Sequence')
plt.legend()
plt.grid(True)
plt.show()

print("Best Sequence of Hidden States:", best_sequence)
