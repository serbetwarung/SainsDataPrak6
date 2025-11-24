# dataset_loader.py
import tensorflow as tf
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split

class BrainTumorDataLoader:
    def __init__(self, data_path="brain_tumor_dataset", img_size=(224, 224)):
        self.data_path = data_path
        self.img_size = img_size
        self.class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.class_dict = {name: idx for idx, name in enumerate(self.class_names)}
    
    def load_data_from_folders(self):
        """Load data dari folder Training dan Testing"""
        print("📁 Loading data from folders...")
        
        # Load training data
        X_train, y_train = self._load_from_folder('Training')
        # Load testing data  
        X_test, y_test = self._load_from_folder('Testing')
        
        print(f"✅ Training data: {X_train.shape}, {y_train.shape}")
        print(f"✅ Testing data: {X_test.shape}, {y_test.shape}")
        
        return (X_train, y_train), (X_test, y_test)
    
    def _load_from_folder(self, folder_name):
        """Load images dari folder tertentu"""
        images = []
        labels = []
        
        folder_path = os.path.join(self.data_path, folder_name)
        
        for class_name in self.class_names:
            class_path = os.path.join(folder_path, class_name)
            if not os.path.exists(class_path):
                print(f"⚠️ Warning: Folder {class_path} tidak ditemukan")
                continue
                
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_path, img_file)
                    
                    try:
                        # Load dan preprocess image
                        img = cv2.imread(img_path)
                        if img is None:
                            continue
                            
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, self.img_size)
                        img = img.astype('float32') / 255.0
                        
                        images.append(img)
                        labels.append(self.class_dict[class_name])
                        
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
                        continue
        
        return np.array(images), tf.keras.utils.to_categorical(labels, num_classes=4)
    
    def get_data_generators(self, batch_size=32, validation_split=0.2):
        """Membuat data generators dengan augmentation untuk training"""
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            validation_split=validation_split
        )
        
        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
        
        train_generator = train_datagen.flow_from_directory(
            os.path.join(self.data_path, 'Training'),
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_directory(
            os.path.join(self.data_path, 'Training'),
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        print(f"✅ Training samples: {train_generator.samples}")
        print(f"✅ Validation samples: {val_generator.samples}")
        
        return train_generator, val_generator

# Contoh penggunaan
if __name__ == "__main__":
    loader = BrainTumorDataLoader()
    train_gen, val_gen = loader.get_data_generators()
    print("Data loader siap!")