import os
import json
import logging
import cv2
import numpy as np
import mediapipe as mp
import requests
import tempfile
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import LabelEncoder

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def load_dataset(filepath):
    with open(filepath, 'r') as f:
        dataset = json.load(f)
    logging.info(f"Loaded dataset from {filepath}")
    return dataset

def process_frame(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return extract_keypoints(results)

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def process_video(url, max_frames=30):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_path = temp_file.name

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            logging.error(f"Failed to open video file from {url}")
            return np.zeros((max_frames, 1662))
        
        frames_processed = 0
        all_frame_keypoints = []

        while frames_processed < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            keypoints = process_frame(frame)
            all_frame_keypoints.append(keypoints)
            
            frames_processed += 1

        cap.release()
        os.remove(temp_path)

        if len(all_frame_keypoints) < max_frames:
            all_frame_keypoints.extend([np.zeros(1662)] * (max_frames - len(all_frame_keypoints)))
        elif len(all_frame_keypoints) > max_frames:
            all_frame_keypoints = all_frame_keypoints[:max_frames]

        return np.array(all_frame_keypoints)

    except Exception as e:
        logging.error(f"Error processing video {url}: {e}")
        return np.zeros((max_frames, 1662))

def process_dataset(dataset):
    X = []
    y = []
    for item in tqdm(dataset, desc="Processing videos"):
        features = process_video(item['url'])
        X.append(features)
        y.append(item['word'])
    return np.array(X), np.array(y)

def visualize_sample(X, y, index):
    sample = X[index]
    label = y[index]
    
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle(f"Sample visualization for '{label}'")
    
    # Plot first, middle, and last frames
    frames_to_plot = [0, len(sample)//2, -1]
    for i, frame_idx in enumerate(frames_to_plot):
        frame = sample[frame_idx]
        
        # Pose
        ax = fig.add_subplot(3, 4, i*4 + 1)
        ax.imshow(frame[:33*4].reshape(33, 4))
        ax.set_title(f"Pose (Frame {frame_idx})")
        
        # Face
        ax = fig.add_subplot(3, 4, i*4 + 2)
        ax.imshow(frame[33*4:33*4+468*3].reshape(468, 3))
        ax.set_title(f"Face (Frame {frame_idx})")
        
        # Left Hand
        ax = fig.add_subplot(3, 4, i*4 + 3)
        ax.imshow(frame[33*4+468*3:33*4+468*3+21*3].reshape(21, 3))
        ax.set_title(f"Left Hand (Frame {frame_idx})")
        
        # Right Hand
        ax = fig.add_subplot(3, 4, i*4 + 4)
        ax.imshow(frame[33*4+468*3+21*3:].reshape(21, 3))
        ax.set_title(f"Right Hand (Frame {frame_idx})")
    
    # Full sequence
    ax = fig.add_subplot(3, 4, (9, 12))
    ax.imshow(sample.T)
    ax.set_title("Full Sequence")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Keypoint")
    
    # 3D trajectory of key points
    ax = fig.add_subplot(3, 4, (5, 8), projection='3d')
    key_points = [
        (33*4, "Nose"),  # Nose from pose
        (33*4 + 468*3, "Left Wrist"),  # Left wrist
        (33*4 + 468*3 + 21*3, "Right Wrist")  # Right wrist
    ]
    
    for point_idx, point_name in key_points:
        x = sample[:, point_idx]
        y = sample[:, point_idx + 1]
        z = sample[:, point_idx + 2]
        ax.plot(x, y, z, label=point_name)
    
    ax.set_title("3D Trajectory of Key Points")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"sample_visualization_{label}.png")
    plt.close()

def main():
    output_dir = 'asl_data'
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = load_dataset(os.path.join('asl_dataset.json'))
    X, y = process_dataset(dataset)
    
    np.save(os.path.join(output_dir, 'X.npy'), X)
    np.save(os.path.join(output_dir, 'y.npy'), y)
    
    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    np.save(os.path.join(output_dir, 'label_encoder.npy'), label_encoder.classes_)
    
    logging.info(f"Processed data shape: {X.shape}")
    logging.info(f"Labels shape: {y.shape}")
    logging.info(f"Number of unique classes: {len(label_encoder.classes_)}")
    
    # Visualize samples for 10 random words
    unique_words = np.unique(y)
    selected_words = np.random.choice(unique_words, min(10, len(unique_words)), replace=False)
    for word in selected_words:
        word_indices = np.where(y == word)[0]
        for _ in range(2):  # Visualize 2 samples per word
            if len(word_indices) > 0:
                index = np.random.choice(word_indices)
                visualize_sample(X, y, index)

if __name__ == "__main__":
    main()