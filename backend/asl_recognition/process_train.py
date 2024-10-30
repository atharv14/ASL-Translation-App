from collections import Counter
import os
import pickle
import json
import cv2
import numpy as np
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import random
import h5py
import tempfile

# Set the base data path
DATA_PATH = "processed_data/"
TEMP_PATH = tempfile.gettempdir()

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# MediaPipe initialization
mp_holistic = mp.solutions.holistic

# Function to perform MediaPipe detection
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Function to extract keypoints
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# Function to process videos and extract keypoints
def process_videos(data_path, actions, save_interval=100):
    sequences = []
    labels = []
    processed_count = 0
    
    checkpoint_path = os.path.join(DATA_PATH, 'video_processing_checkpoint.pkl')
    
    # Check if there's a checkpoint
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        sequences = checkpoint['sequences']
        labels = checkpoint['labels']
        processed_count = checkpoint['processed_count']
        print(f"Resuming from checkpoint. {processed_count} videos already processed.")
    
    for action_idx, action in enumerate(actions[processed_count // len(data_path[actions[0]]):], start=processed_count // len(data_path[actions[0]])):
        for video_idx, video_url in enumerate(data_path[action][processed_count % len(data_path[actions[0]]):], start=processed_count % len(data_path[actions[0]])):
            cap = cv2.VideoCapture(video_url)
            frames = []
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    image, results = mediapipe_detection(frame, holistic)
                    keypoints = extract_keypoints(results)
                    frames.append(keypoints)
                
                if frames:  # Only add if frames were successfully extracted
                    sequences.append(frames)
                    labels.append(action_idx)
            cap.release()
            
            processed_count += 1
            if processed_count % save_interval == 0:
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump({
                        'sequences': sequences,
                        'labels': labels,
                        'processed_count': processed_count
                    }, f)
                print(f"Checkpoint saved. Processed {processed_count} videos.")
    
    # Save final results
    with open(os.path.join(DATA_PATH, 'sequences.pkl'), 'wb') as f:
        pickle.dump(sequences, f)
    np.save(os.path.join(DATA_PATH, 'labels.npy'), labels)
    print("Video processing completed. Results saved to sequences.pkl and labels.npy")
    return sequences, np.array(labels)

# def augment_sequence(sequence, num_augmentations=1):
#     augmented_sequences = []
#     for _ in range(num_augmentations):
#         # Time warping
#         factor = np.random.uniform(0.8, 1.2)
#         new_len = int(len(sequence) * factor)
#         indices = np.linspace(0, len(sequence) - 1, new_len)
#         augmented_sequence = [np.interp(indices, range(len(sequence)), np.array(sequence)[:, i]) 
#                               for i in range(len(sequence[0]))]
#         augmented_sequences.append(list(map(list, zip(*augmented_sequence))))
#     return augmented_sequences

def custom_train_test_split(labels, test_size=0.2, random_state=42):
    random.seed(random_state)
    np.random.seed(random_state)

    # Count samples per class
    class_counts = Counter(labels)
    
    train_indices = []
    test_indices = []
    
    for class_label, count in class_counts.items():
        indices = np.where(labels == class_label)[0]
        np.random.shuffle(indices)
        
        if count == 1:
            # If only one sample, put it in train
            train_indices.extend(indices)
        elif count == 2:
            # If two samples, put one in train, one in test
            train_indices.append(indices[0])
            test_indices.append(indices[1])
        else:
            # Otherwise, split according to test_size
            n_test = max(1, int(count * test_size))
            test_indices.extend(indices[:n_test])
            train_indices.extend(indices[n_test:])
    
    return train_indices, test_indices


def load_data():
    try:
        with open(os.path.join(DATA_PATH, 'sequences.pkl'), 'rb') as f:
            sequences = pickle.load(f)
        print(f"Number of sequences: {len(sequences)}")
    except FileNotFoundError:
        print("sequences.pkl file not found. Please run the video processing step first.")
        return None, None
    except Exception as e:
        print(f"Error loading sequences: {e}")
        return None, None

    try:
        labels = np.load(os.path.join(DATA_PATH, 'labels.npy'))
        print(f"Number of labels: {len(labels)}")
    except FileNotFoundError:
        print("labels.npy file not found. Please run the video processing step first.")
        return sequences, None
    except Exception as e:
        print(f"Error loading labels: {e}")
        return sequences, None

    print("Sequences and Labels are loaded.")
    return sequences, labels

def process_data(chunk_size=100, test_size=0.2, random_state=42):
    sequences, labels = load_data()
    if sequences is None or labels is None:
        print("Data loading failed. Exiting process_data function.")
        return
    
    # Adjust labels to ensure they start from 0
    unique_labels = np.unique(labels)
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    adjusted_labels = np.array([label_map[label] for label in labels])

    num_classes = len(unique_labels)
    max_length = max(len(seq) for seq in sequences)
    feature_dim = len(sequences[0][0])
    class_distribution = Counter(adjusted_labels)
    
    print(f"Number of unique classes: {num_classes}")
    print(f"Max sequence length: {max_length}")
    print(f"Feature dimension: {feature_dim}")
    print("Class distribution:")
    print(class_distribution)
    
    train_indices, test_indices = custom_train_test_split(adjusted_labels, test_size=test_size, random_state=random_state)
    
    # Create HDF5 files for train and test data
    with h5py.File(os.path.join(TEMP_PATH, 'train_data.h5'), 'w') as train_file, \
         h5py.File(os.path.join(TEMP_PATH, 'test_data.h5'), 'w') as test_file:
        
        # Create datasets in the HDF5 files
        train_file.create_dataset('X', shape=(len(train_indices), max_length, feature_dim), 
                                  maxshape=(None, max_length, feature_dim), chunks=True, dtype='float32')
        train_file.create_dataset('y', shape=(len(train_indices), num_classes), 
                                  maxshape=(None, num_classes), chunks=True, dtype='float32')
        
        test_file.create_dataset('X', shape=(len(test_indices), max_length, feature_dim), 
                                 maxshape=(None, max_length, feature_dim), chunks=True, dtype='float32')
        test_file.create_dataset('y', shape=(len(test_indices), num_classes), 
                                 maxshape=(None, num_classes), chunks=True, dtype='float32')

        
        def process_chunk(indices, is_train):
            X_chunk = pad_sequences([sequences[idx] for idx in indices], maxlen=max_length, dtype='float32', padding='post', value=0.0)
            y_chunk = to_categorical([adjusted_labels[idx] for idx in indices], num_classes=num_classes)
            
            file = train_file if is_train else test_file
            start_idx = file['X'].shape[0] - len(indices)
            file['X'][start_idx:] = X_chunk
            file['y'][start_idx:] = y_chunk
        
        for i in range(0, len(train_indices), chunk_size):
            process_chunk(train_indices[i:i+chunk_size], True)
            print(f"Processed and saved train chunk {i//chunk_size + 1}")
        
        for i in range(0, len(test_indices), chunk_size):
            process_chunk(test_indices[i:i+chunk_size], False)
            print(f"Processed and saved test chunk {i//chunk_size + 1}")
    
    # Save metadata
    with open(os.path.join(DATA_PATH, 'asl_metadata.pkl'), 'wb') as f:
        pickle.dump({
            'max_length': max_length,
            'num_classes': num_classes,
            'feature_dim': feature_dim,
            'class_distribution': class_distribution,
            'train_samples': len(train_indices),
            'test_samples': len(test_indices),
            'label_map': label_map
        }, f)
    
    print("Data processing completed.")
    print(f"Train data saved to: {os.path.join(TEMP_PATH, 'train_data.h5')}")
    print(f"Test data saved to: {os.path.join(TEMP_PATH, 'test_data.h5')}")

# Main execution
if __name__ == "__main__":
    
    # Load and prepare the dataset
    with open(os.path.join(DATA_PATH, 'asl_dataset.json'), 'r') as f:
        data = json.load(f)

    actions = list(set([item['word'] for item in data]))
    data_path = {action: [item['url'] for item in data if item['word'] == action] for action in actions}

    # Process videos and extract keypoints
    if not (os.path.exists(os.path.join(DATA_PATH, 'sequences.pkl')) and 
            os.path.exists(os.path.join(DATA_PATH, 'labels.npy'))):
        process_videos(data_path, actions)

    # Print debug information
    with open(os.path.join(DATA_PATH, 'sequences.pkl'), 'rb') as f:
        sequences = pickle.load(f)
    print(f"Number of sequences: {len(sequences)}")
    print(f"Number of labels: {len(np.load(os.path.join(DATA_PATH, 'labels.npy')))}")
    print(f"Actions: {actions}")
    print(f"Label distribution: {np.bincount(np.load(os.path.join(DATA_PATH, 'labels.npy')))}")

    # Call the main processing function
    process_data()
