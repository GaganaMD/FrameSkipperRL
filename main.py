"""Main script for Auto-Skipper project."""

import os
import numpy as np
from video_processor import VideoProcessor
from environment import VideoSkipEnvironment
from dqn_model import DQNAgent
from config import TRAINING_CONFIG, PATHS, DQN_CONFIG

from utils.helpers import create_directories, log_message

class AutoSkipper:
    def __init__(self):
        create_directories()
        self.video_processor = VideoProcessor()
        self.environment = VideoSkipEnvironment()
        self.agent = DQNAgent(
            state_size=self.environment.observation_space.shape[0],
            action_size=self.environment.action_space.n
        )
        
        log_message("Auto-Skipper initialized")
    
    def train(self, episodes=None):
        """Train the DQN agent."""
        episodes = episodes or TRAINING_CONFIG['episodes']
        scores = []
        
        for episode in range(episodes):
            state = self.environment.reset()
            total_reward = 0
            steps = 0
            
            while steps < TRAINING_CONFIG['max_steps_per_episode']:
                action = self.agent.act(state)
                next_state, reward, done, info = self.environment.step(action)
                
                self.agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    break
                
                # Train agent
                if len(self.agent.memory) > self.agent.batch_size:
                    self.agent.replay()
            
            scores.append(total_reward)
            
            # Update target network
            if episode % DQN_CONFIG['target_update'] == 0:
                self.agent.update_target_network()
            
            # Logging
            if episode % TRAINING_CONFIG['log_frequency'] == 0:
                avg_score = np.mean(scores[-TRAINING_CONFIG['log_frequency']:])
                log_message(f"Episode {episode}, Average Score: {avg_score:.2f}, Epsilon: {self.agent.epsilon:.3f}")
            
            # Save model
            if episode % TRAINING_CONFIG['save_frequency'] == 0 and episode > 0:
                model_path = os.path.join(PATHS['models_dir'], f'model_episode_{episode}')
                self.agent.save_model(model_path)
        
        log_message(f"Training completed after {episodes} episodes")
        return scores
    
    def process_video(self, video_path):
        """Process a video and return skip recommendations."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Detect intro/outro segments
        detection_results = self.video_processor.detect_intro_outro(video_path)
        
        # Use trained agent for additional decisions (simplified for demo)
        recommendations = {
            'intro_segments': detection_results['intro_segments'],
            'outro_segments': detection_results['outro_segments'],
            'confidence_scores': []
        }
        
        log_message(f"Processed video: {video_path}")
        return recommendations
    
    def demo_mode(self, video_path):
        """Run demo with a sample video."""
        print("=== Auto-Skipper Demo ===")
        print(f"Processing: {video_path}")
        
        try:
            results = self.process_video(video_path)
            
            print("\n--- Detection Results ---")
            if results['intro_segments']:
                for start, end in results['intro_segments']:
                    print(f"Intro detected: {start:.1f}s - {end:.1f}s")
            
            if results['outro_segments']:
                for start, end in results['outro_segments']:
                    print(f"Outro detected: {start:.1f}s - {end:.1f}s")
            
            if not results['intro_segments'] and not results['outro_segments']:
                print("No intro/outro segments detected")
            
        except Exception as e:
            print(f"Error processing video: {e}")

def main():
    """Main function."""
    skipper = AutoSkipper()
    
    # Check for sample videos
    sample_dir = PATHS['sample_videos_dir']
    if not os.path.exists(sample_dir):
        print(f"Please add sample videos to {sample_dir}")
        return
    
    # Training mode
    print("Starting training...")
    scores = skipper.train(episodes=100)  # Quick training for demo
    
    # Demo mode
    sample_videos = [f for f in os.listdir(sample_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    if sample_videos:
        sample_video = os.path.join(sample_dir, sample_videos[0])
        skipper.demo_mode(sample_video)
    else:
        print("No sample videos found for demo")

if __name__ == "__main__":
    main()
