"""Configuration settings for Auto-Skipper project."""

# DQN Settings
DQN_CONFIG = {
    'learning_rate': 0.001,
    'epsilon_start': 1.0,
    'epsilon_end': 0.1,
    'epsilon_decay': 0.995,
    'batch_size': 32,
    'memory_size': 10000,
    'target_update': 100,
    'hidden_layers': [128, 64],
}

# Video Processing Settings
VIDEO_CONFIG = {
    'frame_rate': 1,
    'resize_dims': (224, 224),
    'max_duration': 3600,
}

# Feature Extraction Settings
FEATURE_CONFIG = {
    'histogram_bins': 32,
    'edge_threshold': 100,
    'similarity_threshold': 0.8,
    'window_size': 30,
}

# Training Settings
TRAINING_CONFIG = {
    'episodes': 1000,
    'max_steps_per_episode': 1000,
    'save_frequency': 100,
    'log_frequency': 10,
}

# File Paths
PATHS = {
    'models_dir': 'models/',
    'logs_dir': 'logs/',
    'data_dir': 'data/',
    'sample_videos_dir': 'data/sample_videos/',
}
