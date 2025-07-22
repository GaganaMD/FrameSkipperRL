"""Utility functions for the Auto-Skipper project."""

import os
import json
import numpy as np
from datetime import datetime

def create_directories():
    """Create necessary directories if they don't exist."""
    dirs = ['models', 'logs', 'data/sample_videos']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print("Directories created successfully!")

def log_message(message, log_file='logs/training.log'):
    """Log messages with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}\n"
    
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, 'a') as f:
        f.write(log_entry)
    print(log_entry.strip())

def save_json(data, filepath):
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(filepath):
    """Load data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def normalize_features(features):
    """Normalize feature vectors."""
    return (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)
