"""Custom environment for DQN training."""

import gym
from gym import spaces
import numpy as np
from video_processor import VideoProcessor
from utils.helpers import log_message

class VideoSkipEnvironment(gym.Env):
    def __init__(self, video_data=None):
        super(VideoSkipEnvironment, self).__init__()
        
        # Action space: 0 = Continue watching, 1 = Skip
        self.action_space = spaces.Discrete(2)
        
        # Observation space: Features + context
        self.feature_dim = 97  # Color histogram (96) + edge density (1)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.feature_dim + 3,), dtype=np.float32  # +3 for context
        )
        
        self.video_processor = VideoProcessor()
        self.reset()
    
    def reset(self):
        """Reset environment for new episode."""
        # Initialize with dummy data or load real video data
        self.current_step = 0
        self.max_steps = 100
        self.video_duration = 300  # 5 minutes
        self.intro_segments = [(0, 30)]  # Mock intro segment
        self.outro_segments = [(270, 300)]  # Mock outro segment
        
        # Generate random features for first state
        base_features = np.random.random(self.feature_dim)
        context = [
            self.current_step / self.max_steps,  # Progress
            0,  # In intro segment
            0   # In outro segment
        ]
        
        self.state = np.concatenate([base_features, context])
        return self.state
    
    def step(self, action):
        """Execute one step in environment."""
        current_time = (self.current_step / self.max_steps) * self.video_duration
        
        # Determine if we're in intro/outro segment
        in_intro = any(start <= current_time <= end for start, end in self.intro_segments)
        in_outro = any(start <= current_time <= end for start, end in self.outro_segments)
        
        # Calculate reward
        reward = self._calculate_reward(action, in_intro, in_outro)
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # Generate next state
        base_features = np.random.random(self.feature_dim)
        context = [
            self.current_step / self.max_steps,  # Progress
            1 if in_intro else 0,  # In intro segment
            1 if in_outro else 0   # In outro segment
        ]
        
        self.state = np.concatenate([base_features, context])
        
        info = {
            'current_time': current_time,
            'in_intro': in_intro,
            'in_outro': in_outro,
            'action_taken': 'skip' if action == 1 else 'continue'
        }
        
        return self.state, reward, done, info
    
    def _calculate_reward(self, action, in_intro, in_outro):
        """Calculate reward based on action and context."""
        if action == 1:  # Skip action
            if in_intro or in_outro:
                return 1.0  # Positive reward for skipping intro/outro
            else:
                return -0.5  # Negative reward for skipping content
        else:  # Continue action
            if in_intro or in_outro:
                return -0.1  # Small negative reward for not skipping
            else:
                return 0.1  # Small positive reward for watching content
    
    def render(self, mode='human'):
        """Render environment state."""
        current_time = (self.current_step / self.max_steps) * self.video_duration
        print(f"Step: {self.current_step}, Time: {current_time:.1f}s")
