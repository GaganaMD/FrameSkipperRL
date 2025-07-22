"""Deep Q-Network implementation for Auto-Skipper."""

import numpy as np
import tensorflow as tf
from collections import deque
import random
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import DQN_CONFIG

from utils.helpers import log_message, save_json

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=DQN_CONFIG['memory_size'])
        
        # Hyperparameters
        self.epsilon = DQN_CONFIG['epsilon_start']
        self.epsilon_min = DQN_CONFIG['epsilon_end']
        self.epsilon_decay = DQN_CONFIG['epsilon_decay']
        self.learning_rate = DQN_CONFIG['learning_rate']
        self.batch_size = DQN_CONFIG['batch_size']
        
        # Neural networks
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        self.update_target_network()
        
        log_message("DQN Agent initialized")
    
    def _build_model(self):
        """Build neural network model."""
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(DQN_CONFIG['hidden_layers'][0], 
                                       input_dim=self.state_size, activation='relu'))
        model.add(tf.keras.layers.Dense(DQN_CONFIG['hidden_layers'][1], activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate),
                     loss='mse')
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose action using epsilon-greedy policy."""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        q_values = self.q_network.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self):
        """Train the model on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        # Current Q-values
        current_q_values = self.q_network.predict(states, verbose=0)
        
        # Next Q-values from target network
        next_q_values = self.target_network.predict(next_states, verbose=0)
        
        # Update Q-values
        for i in range(self.batch_size):
            if dones[i]:
                current_q_values[i][actions[i]] = rewards[i]
            else:
                current_q_values[i][actions[i]] = rewards[i] + 0.95 * np.max(next_q_values[i])
        
        # Train the model
        self.q_network.fit(states, current_q_values, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """Update target network weights."""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def save_model(self, filepath):
        """Save the trained model."""
        self.q_network.save(f"{filepath}_main.h5")
        self.target_network.save(f"{filepath}_target.h5")
        
        # Save agent parameters
        params = {
            'epsilon': self.epsilon,
            'state_size': self.state_size,
            'action_size': self.action_size
        }
        save_json(params, f"{filepath}_params.json")
        
        log_message(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model."""
        self.q_network = tf.keras.models.load_model(f"{filepath}_main.h5")
        self.target_network = tf.keras.models.load_model(f"{filepath}_target.h5")
        
        # Load agent parameters
        import json
        with open(f"{filepath}_params.json", 'r') as f:
            params = json.load(f)
        self.epsilon = params['epsilon']
        
        log_message(f"Model loaded from {filepath}")
