import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, TimeDistributed
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
import shap

def create_lstm_model(hp):
    model = Sequential()
    model.add(Bidirectional(LSTM(hp.Int('lstm_units', 32, 256, step=32),
                                 return_sequences=True,
                                 input_shape=(None, 1662))))
    model.add(Dropout(hp.Float('dropout_1', 0, 0.5, step=0.1)))
    model.add(Bidirectional(LSTM(hp.Int('lstm_units_2', 32, 256, step=32),
                                 return_sequences=False)))
    model.add(Dropout(hp.Float('dropout_2', 0, 0.5, step=0.1)))
    model.add(Dense(hp.Int('dense_units', 32, 256, step=32), activation='relu'))
    model.add(Dropout(hp.Float('dropout_3', 0, 0.5, step=0.1)))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(
        optimizer=Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG')),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    return model

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

def plot_shap_values(model, X_test):
    explainer = shap.DeepExplainer(model, X_test[:100])
    shap_values = explainer.shap_values(X_test[:10])
    shap.summary_plot(shap_values, X_test[:10], plot_type="bar")
    plt.savefig('shap_summary.png')
    plt.close()

def main():
    # Load preprocessed data
    X_train = np.load('X_train.npy', mmap_mode='r')
    y_train = np.load('y_train.npy', mmap_mode='r')
    X_test = np.load('X_test.npy', mmap_mode='r')
    y_test = np.load('y_test.npy', mmap_mode='r')
    
    with open('asl_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    num_classes = metadata['num_classes']
    
    # Hyperparameter tuning
    tuner = RandomSearch(
        create_lstm_model,
        objective='val_categorical_accuracy',
        max_trials=5,
        executions_per_trial=1,
        directory='hyperparameter_tuning',
        project_name='asl_recognition'
    )
    
    tuner.search(X_train, y_train,
                 epochs=50,
                 validation_split=0.2,
                 callbacks=[EarlyStopping(patience=5)])
    
    best_model = tuner.get_best_models(num_models=1)[0]
    
    # Train the best model
    history = best_model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[
            TensorBoard(log_dir='Logs'),
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-6)
        ]
    )
    
    # Evaluate the model
    test_loss, test_accuracy = best_model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Generate predictions
    y_pred = best_model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true_classes, y_pred_classes, range(num_classes))
    
    # Plot SHAP values
    plot_shap_values(best_model, X_test)
    
    # Save the model
    best_model.save('asl_recognition_best_model.keras')
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(history.history['categorical_accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_categorical_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(122)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    print("Training completed. Model and visualizations saved.")

if __name__ == "__main__":
    main()