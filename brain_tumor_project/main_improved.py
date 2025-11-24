# main_improved.py
import tensorflow as tf
import os
import json
import numpy as np
from datetime import datetime

# Import modul custom
from dataset_loader import BrainTumorDataLoader
from model_cnn import create_cnn_model, compile_cnn_model, create_improved_cnn_model
from model_resnet50 import create_resnet50_model, compile_resnet50_model, create_enhanced_resnet50
from model_vit import create_vit_model, compile_vit_model, create_enhanced_vit_model
from utils import TrainingUtils, GradCAM, check_gpu

class EnhancedTraining:
    def __init__(self):
        self.data_loader = BrainTumorDataLoader("brain_tumor_dataset", (224, 224))
        self.class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    def setup_enhanced_data_augmentation(self):
        """Enhanced data augmentation"""
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,  # Increased from 20
            width_shift_range=0.3,  # Increased from 0.2
            height_shift_range=0.3,
            shear_range=0.3,  # Increased from 0.2
            zoom_range=0.3,   # Increased from 0.2
            horizontal_flip=True,
            vertical_flip=True,  # Added vertical flip
            brightness_range=[0.7, 1.3],  # Wider range
            channel_shift_range=0.2,  # New: color augmentation
            fill_mode='nearest',
            validation_split=0.2
        )
        
        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
        
        return train_datagen, val_datagen
    
    def create_enhanced_callbacks(self, model_name):
        """Enhanced callbacks dengan learning rate scheduling"""
        os.makedirs('saves_enhanced', exist_ok=True)
        
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',  # Changed to accuracy
                patience=15,  # Increased patience
                restore_best_weights=True,
                verbose=1,
                mode='max'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,  # Increased patience
                min_lr=1e-7,
                verbose=1,
                mode='min'
            ),
            tf.keras.callbacks.ModelCheckpoint(
                f'saves_enhanced/best_{model_name}.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1,
                mode='max'
            ),
            tf.keras.callbacks.CSVLogger(
                f'saves_enhanced/training_{model_name}.csv',
                separator=',',
                append=False
            ),
            # Tambah TensorBoard untuk monitoring
            tf.keras.callbacks.TensorBoard(
                log_dir=f'saves_enhanced/logs_{model_name}',
                histogram_freq=1
            )
        ]
    
    def calculate_class_weights(self):
        """Calculate class weights untuk handle imbalance"""
        train_gen, _ = self.data_loader.get_data_generators(batch_size=32)
        class_counts = np.bincount(train_gen.classes)
        total_samples = np.sum(class_counts)
        num_classes = len(class_counts)
        
        class_weights = {}
        for i in range(num_classes):
            class_weights[i] = total_samples / (num_classes * class_counts[i])
        
        print(f"🎯 Class weights: {class_weights}")
        return class_weights

def main_enhanced_training():
    print("🚀 ENHANCED TRAINING - OPTION 1-3 IMPLEMENTATION")
    print("=" * 60)
    
    # Setup
    check_gpu()
    enhancer = EnhancedTraining()
    
    # Parameters
    BATCH_SIZE = 32
    EPOCHS = 30  # Increased from 5
    
    # Enhanced data generators
    print("\n📁 SETTING UP ENHANCED DATA PIPELINE...")
    train_datagen, val_datagen = enhancer.setup_enhanced_data_augmentation()
    
    train_gen = train_datagen.flow_from_directory(
        "brain_tumor_dataset/Training",
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    val_gen = val_datagen.flow_from_directory(
        "brain_tumor_dataset/Training", 
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    # Class weights untuk handle imbalance
    class_weights = enhancer.calculate_class_weights()
    
    # Enhanced Models
    print("\n🏗️ CREATING ENHANCED MODELS...")
    
    # 1. Enhanced CNN (Fix Overfitting)
    print("🔸 Creating Enhanced CNN...")
    cnn_model = create_improved_cnn_model()
    cnn_model = compile_cnn_model(cnn_model, learning_rate=0.001)
    
    # 2. Enhanced ResNet50
    print("🔸 Creating Enhanced ResNet50...")
    resnet_model = create_enhanced_resnet50()
    resnet_model = compile_resnet50_model(resnet_model, learning_rate=0.0001)
    
    # 3. Enhanced ViT
    print("🔸 Creating Enhanced ViT...")
    vit_model = create_enhanced_vit_model()
    vit_model = compile_vit_model(vit_model, learning_rate=0.0001)
    
    models = {
        'CNN_Enhanced': cnn_model,
        'ResNet50_Enhanced': resnet_model, 
        'ViT_Enhanced': vit_model
    }
    
    # Training dengan enhancement
    print(f"\n🎯 STARTING ENHANCED TRAINING ({EPOCHS} EPOCHS)...")
    history_dict = {}
    
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"TRAINING {name}")
        print(f"{'='*50}")
        
        callbacks = enhancer.create_enhanced_callbacks(name)
        
        history = model.fit(
            train_gen,
            epochs=EPOCHS,
            validation_data=val_gen,
            callbacks=callbacks,
            class_weight=class_weights,  # Added class weights
            verbose=1
        )
        
        history_dict[name] = history
        print(f"✅ {name} training completed!")
    
    print(f"\n🎉 ENHANCED TRAINING COMPLETED!")
    return history_dict

if __name__ == "__main__":
    main_enhanced_training()