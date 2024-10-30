import tensorflow as tf
import cv2
import numpy as np
import json
from io import BytesIO

class ASLRecognitionModel:
    def __init__(self, model_path, classes_path):
        self.model = tf.keras.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.classes = json.load(f)
        self.class_names = {v: k for k, v in self.classes.items()}

    def extract_frames(self, video_content, num_frames=16):
        frames = []
        
        if isinstance(video_content, str):  # If it's a file path
            cap = cv2.VideoCapture(video_content)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()
        else:  # If it's BytesIO content
            video_array = np.frombuffer(video_content.getvalue(), dtype=np.uint8)
            i = 0
            while True:
                try:
                    frame = cv2.imdecode(video_array[i:], cv2.IMREAD_COLOR)
                    if frame is None:
                        break
                    frames.append(frame)
                    i += len(frame.tobytes())
                except cv2.error:
                    break
        
        if len(frames) > num_frames:
            frame_indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
            frames = [frames[i] for i in frame_indices]
        elif len(frames) < num_frames:
            last_frame = frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8)
            frames.extend([last_frame] * (num_frames - len(frames)))
        
        frames = [cv2.resize(frame, (224, 224)) for frame in frames]
        
        return np.array([frames])

    def recognize(self, video_content):
        preprocessed_video = self.extract_frames(video_content)
        predictions = self.model.predict(preprocessed_video)
        class_index = np.argmax(predictions[0])
        return self.class_names[class_index]

model_path = 'asl_recognition_model.keras'
classes_path = 'asl_classes.json'
model = ASLRecognitionModel(model_path, classes_path)

def recognize_asl(video_content):
    return model.recognize(video_content)

def test_local_video(video_path):
    result = recognize_asl(video_path)
    print(f"The recognized ASL sign in '{video_path}' is: {result}")

if __name__ == "__main__":
    # Test with local video files
    test_videos = ["match.mp4", "hello.mp4", "world.mp4", "no.mp4", "download.mp4"]
    for video in test_videos:
        print(f"\nTesting {video}:")
        test_local_video(video)