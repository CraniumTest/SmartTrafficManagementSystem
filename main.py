import os
import traci
import traci.constants as tc
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random
import matplotlib.pyplot as plt

# Constants
MAX_EPISODES = 1000
MAX_STEPS = 1000
GAMMA = 0.95
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
LEARNING_RATE = 0.001
MEMORY_SIZE = 2000
BATCH_SIZE = 32

# SIMULATION SETUP
SUMO_BINARY = "sumo"  # The SUMO command
CONFIG_FILE = "sumo_config.sumocfg"

class TrafficEnvironment:
    def __init__(self):
        # Setup the SUMO environment
        traci.start([SUMO_BINARY, "-c", CONFIG_FILE])
        self.num_states = 10  # Hypothetical size of the state space
        self.num_actions = 4  # Number of traffic light phases

    def reset(self):
        traci.load(["-c", CONFIG_FILE])
        return self._get_state()

    def step(self, action):
        self._set_signal(action)
        for _ in range(5):
            traci.simulationStep()
        state = self._get_state()
        reward = self._get_reward()
        done = traci.simulation.getTime() > MAX_STEPS  # Placeholder condition
        return state, reward, done

    def _get_state(self):
        # Simulated state extraction
        return np.random.random(self.num_states)

    def _set_signal(self, action):
        # Placeholder for real traffic light setting logic
        pass

    def _get_reward(self):
        # Placeholder for real reward calculation
        return -traci.edge.getLastStepHaltingNumber("e1")

    def close(self):
        traci.close()

class DQNAgent:
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON
        self.model = self._build_model(num_states, num_actions)

    def _build_model(self, num_states, num_actions):
        model = keras.Sequential([
            keras.layers.Dense(24, input_dim=num_states, activation='relu'),
            keras.layers.Dense(24, activation='relu'),
            keras.layers.Dense(num_actions, activation='linear')
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='mse')
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.num_actions)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + GAMMA * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > MIN_EPSILON:
            self.epsilon *= EPSILON_DECAY

def main():
    env = TrafficEnvironment()
    agent = DQNAgent(env.num_states, env.num_actions)

    for e in range(MAX_EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, env.num_states])
        
        for time in range(MAX_STEPS):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, env.num_states])
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                print(f"Episode: {e}/{MAX_EPISODES}, score: {time}, e: {agent.epsilon:.2}")
                break

            if len(agent.memory) > BATCH_SIZE:
                agent.replay(BATCH_SIZE)

    env.close()

if __name__ == "__main__":
    main()
