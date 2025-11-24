# utils.py
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import cv2
import os

class TrainingUtils:
    @staticmethod
    def create_callbacks(model_name):
        """Membuat callbacks untuk training"""
        # Buat folder saves jika belum ada
        os.makedirs('saves', exist_ok=True)
        
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=10, 
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                f'saves/best_{model_name}.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.CSVLogger(
                f'saves/training_{model_name}.csv',
                separator=',',
                append=False
            )
        ]
    
    @staticmethod
    def plot_training_history(history_dict):
        """Plot training history untuk multiple models"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        colors = ['blue', 'orange', 'green', 'red']
        
        for idx, (name, history) in enumerate(history_dict.items()):
            color = colors[idx % len(colors)]
            
            # Accuracy
            axes[0, 0].plot(history.history['accuracy'], 
                           color=color, label=f'{name} Train', linewidth=2)
            axes[0, 0].plot(history.history['val_accuracy'], 
                           color=color, linestyle='--', label=f'{name} Val', linewidth=2)
            
            # Loss
            axes[0, 1].plot(history.history['loss'], 
                           color=color, label=f'{name} Train', linewidth=2)
            axes[0, 1].plot(history.history['val_loss'], 
                           color=color, linestyle='--', label=f'{name} Val', linewidth=2)
        
        # Set titles and labels
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Hide empty subplots
        axes[1, 0].axis('off')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('saves/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def evaluate_model(model, X_test, y_test, class_names):
        """Evaluasi model dan tampilkan metrics"""
        if X_test is None or y_test is None:
            print("❌ Test data tidak tersedia")
            return None
            
        # Predict
        y_pred = model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Classification report
        print("\n📊 Classification Report:")
        print(classification_report(y_true_classes, y_pred_classes, 
                                  target_names=class_names, digits=4))
        
        # Confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('saves/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return y_pred_classes
    
    @staticmethod
    def plot_model_comparison(results):
        """Plot perbandingan akurasi model"""
        if not results:
            print("❌ Tidak ada results untuk diplot")
            return
            
        plt.figure(figsize=(10, 6))
        names = list(results.keys())
        accuracies = [results[name]['test_accuracy'] for name in names]
        
        bars = plt.bar(names, accuracies, color=['blue', 'orange', 'green'], alpha=0.7)
        plt.title('Perbandingan Akurasi Model')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.grid(True, alpha=0.3)
        plt.savefig('saves/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

class GradCAM:
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        try:
            self.grad_model = tf.keras.models.Model(
                [model.inputs], 
                [model.get_layer(layer_name).output, model.output]
            )
        except Exception as e:
            print(f"❌ Error creating GradCAM model: {e}")
            self.grad_model = None
    
    def generate_heatmap(self, image, class_idx=None):
        """Generate Grad-CAM heatmap"""
        if self.grad_model is None:
            return np.zeros((image.shape[0]//32, image.shape[1]//32))
            
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(tf.expand_dims(image, axis=0))
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]
        
        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            return np.zeros((image.shape[0]//32, image.shape[1]//32))
            
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_mean(tf.multiply(conv_outputs, pooled_grads), axis=-1)
        heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)
        
        return heatmap.numpy()
    
    def overlay_heatmap(self, image, heatmap, alpha=0.4):
        """Overlay heatmap on original image"""
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Convert image to uint8
        image_uint8 = (image * 255).astype('uint8')
        
        superimposed_img = heatmap * alpha + image_uint8
        superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
        
        return superimposed_img

def check_gpu():
    """Check GPU availability"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ {len(gpus)} GPU tersedia")
            return True
        except RuntimeError as e:
            print(f"❌ Error configuring GPU: {e}")
            return False
    else:
        print("❌ Tidak ada GPU yang terdeteksi")
        return False

# Test the utils
if __name__ == "__main__":
    print("🧪 Testing utils...")
    print(f"GPU Available: {check_gpu()}")
    print("✅ Utils berhasil diimport!")