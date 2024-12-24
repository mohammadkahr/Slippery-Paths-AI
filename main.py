import numpy as np
from source import FrozenLake  # Import the custom FrozenLake class

# Initialize the environment
env = FrozenLake(render_mode="human", map_name="8x8")
num_states = env.nS  # Total states
num_actions = env.nA  # Total actions

# Parameters
gamma = 0.9  # Discount factor
theta = 1e-3  # Convergence threshold

# Initialize value function and policy
value_function = np.zeros(num_states)  # Value function for all states
policy = np.zeros(num_states, dtype=int)  # Policy initialized to action 0

# Policy Iteration Process
for iteration in range(100):  # Limit iterations

    # Policy Evaluation
    while True:
        delta = 0
        for state in range(num_states):

            # Store old value
            old_value = value_function[state]

            # Retrieve action based on the current policy
            action = policy[state]

            # Calculate new value for this state under current policy
            new_value = 0
            for prob, next_state, reward, done in env.P[state][action]:
                # Extract next_state from tuple if necessary
                if isinstance(next_state, tuple):
                    next_state = next_state[0]
                new_value += prob * (reward + gamma * value_function[next_state])  # Bellman update

            # Update value function and delta
            value_function[state] = new_value
            delta = max(delta, abs(old_value - new_value))

        print(f"Value function after iteration {iteration}: {value_function}")

        if delta < theta:  # Break when converged
            break

    # Policy Improvement
    policy_stable = True
    for state in range(num_states):
        old_action = policy[state]

        # Evaluate all possible actions for this state
        action_values = np.zeros(num_actions)
        for action in range(num_actions):
            for prob, next_state, reward, done in env.P[state][action]:
                if isinstance(next_state, tuple):
                    next_state = next_state[0]
                action_values[action] += prob * (reward + gamma * value_function[next_state])

        # Update policy to action with maximum value
        policy[state] = np.argmax(action_values)

        # Check if policy has changed
        if old_action != policy[state]:
            policy_stable = False

    print(f"Policy after iteration {iteration}: {policy}")

    if policy_stable:  # Stop if policy doesn't change
        print(f"Policy converged after {iteration + 1} iterations.")
        break

# Simulation
state, _ = env.reset()
env.render()

while True:
    action = policy[state]  # Get the action for current state
    next_state, reward, done, _, _ = env.step(action)  # Take action
    next_state = next_state[0] if isinstance(next_state, tuple) else next_state  # Extract next_state
    env.render()  # Render environment state

    print(f"Current state: {state}, Action: {action}, Next state: {next_state}, Reward: {reward}")

    if done:
        print(f"Simulation ended with reward {reward}")
        break

    state = next_state  # Update current state

env.close()  # Clean up
