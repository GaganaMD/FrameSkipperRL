"""Video processing module for Auto-Skipper."""

import cv2
import numpy as np
from config import VIDEO_CONFIG, FEATURE_CONFIG

from utils.helpers import log_message

class VideoProcessor:
    def __init__(self):
        self.frame_rate = VIDEO_CONFIG['frame_rate']
        self.resize_dims = VIDEO_CONFIG['resize_dims']
        self.window_size = FEATURE_CONFIG['window_size']
        
    def load_video(self, video_path):
        """Load video and return basic information."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
        log_message(f"Loaded video: {video_path}, Duration: {duration:.2f}s, FPS: {fps}")
        
        cap.release()
        return {'fps': fps, 'frame_count': frame_count, 'duration': duration}
    
    def extract_frames(self, video_path, start_time=0, end_time=None):
        """Extract frames from video."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        start_frame = int(start_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames = []
        frame_timestamps = []
        frame_idx = start_frame
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            current_time = frame_idx / fps
            if end_time and current_time > end_time:
                break
            
            if frame_idx % self.frame_rate == 0:  # Sample frames
                processed_frame = self.preprocess_frame(frame)
                frames.append(processed_frame)
                frame_timestamps.append(current_time)
            
            frame_idx += 1
        
        cap.release()
        return np.array(frames), frame_timestamps
    
    def preprocess_frame(self, frame):
        """Preprocess individual frame."""
        # Resize frame
        resized = cv2.resize(frame, self.resize_dims)
        # Convert to RGB
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Normalize pixel values
        normalized = rgb_frame.astype(np.float32) / 255.0
        return normalized
    
    def extract_features(self, frames):
        """Extract features from frames."""
        features = []
        
        for frame in frames:
            frame_features = {}
            
            # Convert back to uint8 for OpenCV operations
            frame_uint8 = (frame * 255).astype(np.uint8)
            
            # Color histogram
            hist_r = cv2.calcHist([frame_uint8], [0], None, [FEATURE_CONFIG['histogram_bins']], [0, 256])
            hist_g = cv2.calcHist([frame_uint8], [1], None, [FEATURE_CONFIG['histogram_bins']], [0, 256])
            hist_b = cv2.calcHist([frame_uint8], [2], None, [FEATURE_CONFIG['histogram_bins']], [0, 256])
            
            color_hist = np.concatenate([hist_r.flatten(), hist_g.flatten(), hist_b.flatten()])
            frame_features['color_histogram'] = color_hist
            
            # Edge density
            gray = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, FEATURE_CONFIG['edge_threshold'], 
                             FEATURE_CONFIG['edge_threshold'] * 2)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            frame_features['edge_density'] = edge_density
            
            # Combine all features
            combined_features = np.concatenate([
                color_hist,
                [edge_density]
            ])
            
            features.append(combined_features)
        
        return np.array(features)
    
    def detect_intro_outro(self, video_path):
        """Detect intro and outro segments."""
        frames, timestamps = self.extract_frames(video_path)
        features = self.extract_features(frames)
        
        # Simple pattern detection based on feature similarity
        intro_segments = []
        outro_segments = []
        
        # Look for repetitive patterns at the beginning (intro)
        if len(features) > 30:  # Minimum frames for analysis
            intro_features = features[:30]  # First 30 frames
            intro_similarity = self._calculate_similarity_score(intro_features)
            
            if intro_similarity > FEATURE_CONFIG['similarity_threshold']:
                intro_end_time = timestamps[29] if len(timestamps) > 29 else timestamps[-1]
                intro_segments.append((0, intro_end_time))
                log_message(f"Detected intro: 0 - {intro_end_time:.2f}s")
        
        # Look for repetitive patterns at the end (outro)
        if len(features) > 30:
            outro_features = features[-30:]  # Last 30 frames
            outro_similarity = self._calculate_similarity_score(outro_features)
            
            if outro_similarity > FEATURE_CONFIG['similarity_threshold']:
                outro_start_time = timestamps[-30] if len(timestamps) > 30 else timestamps[0]
                outro_end_time = timestamps[-1]
                outro_segments.append((outro_start_time, outro_end_time))
                log_message(f"Detected outro: {outro_start_time:.2f} - {outro_end_time:.2f}s")
        
        return {
            'intro_segments': intro_segments,
            'outro_segments': outro_segments,
            'features': features,
            'timestamps': timestamps
        }
    
    def _calculate_similarity_score(self, features):
        """Calculate similarity score for a sequence of features."""
        if len(features) < 2:
            return 0.0
        
        similarities = []
        for i in range(len(features) - 1):
            similarity = np.corrcoef(features[i], features[i + 1])[0, 1]
            if not np.isnan(similarity):
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
