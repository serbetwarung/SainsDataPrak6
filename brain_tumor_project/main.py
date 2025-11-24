# main.py
import tensorflow as tf
import os
import json
from datetime import datetime

# Import modul custom
from dataset_loader import BrainTumorDataLoader
from model_cnn import create_cnn_model, compile_cnn_model
from model_resnet50 import create_resnet50_model, compile_resnet50_model
from model_vit import create_vit_model, compile_vit_model
from utils import TrainingUtils, GradCAM, check_gpu

def setup_gpu():
    """Setup GPU configuration"""
    return check_gpu()

def main():
    print("🚀 MEMULAI PRAKTIKUM 6 - BRAIN TUMOR CLASSIFICATION")
    print("=" * 60)
    
    # Setup GPU
    setup_gpu()
    
    # Parameters
    DATA_PATH = "brain_tumor_dataset"
    BATCH_SIZE = 16  # Reduced for stability
    EPOCHS = 5      # Reduced for quick testing
    IMG_SIZE = (224, 224)
    
    # Load data
    print("\n📁 MEMUAT DATASET...")
    data_loader = BrainTumorDataLoader(DATA_PATH, IMG_SIZE)
    
    # Pilih metode loading (gunakan salah satu)
    USE_GENERATORS = True  # Ganti False jika mau load langsung
    
    if USE_GENERATORS:
        # Menggunakan generators (recommended untuk dataset besar)
        train_gen, val_gen = data_loader.get_data_generators(BATCH_SIZE)
        (X_train, y_train), (X_test, y_test) = (None, None), (None, None)
    else:
        # Load langsung ke memory (untuk dataset kecil)
        (X_train, y_train), (X_test, y_test) = data_loader.load_data_from_folders()
        train_gen, val_gen = None, None
    
    class_names = data_loader.class_names
    print(f"🎯 Kelas: {class_names}")
    
    # Buat models
    print("\n🏗️ MEMBUAT MODEL...")
    
    # CNN Model
    print("🔸 Membuat CNN Model...")
    cnn_model = create_cnn_model()
    cnn_model = compile_cnn_model(cnn_model, learning_rate=0.001)
    
    # ResNet50 Model
    print("🔸 Membuat ResNet50 Model...")
    resnet_model = create_resnet50_model()
    resnet_model = compile_resnet50_model(resnet_model, learning_rate=0.0001)
    
    # Vision Transformer Model
    print("🔸 Membuat Vision Transformer Model...")
    vit_model = create_vit_model()
    vit_model = compile_vit_model(vit_model, learning_rate=0.0001)
    
    # Tampilkan summary
    print("\n📋 MODEL SUMMARY:")
    print("CNN Model:")
    cnn_model.summary()
    print(f"\nResNet50 Model (Trainable params: {sum([w.shape.num_elements() for w in resnet_model.trainable_weights])})")
    print(f"\nViT Model (Trainable params: {sum([w.shape.num_elements() for w in vit_model.trainable_weights])})")
    
    # Dictionary untuk models dan histories
    models = {
        'CNN': cnn_model,
        'ResNet50': resnet_model,
        'ViT': vit_model
    }
    
    history_dict = {}
    
    # Training loop
    print(f"\n🎯 MEMULAI TRAINING ({EPOCHS} EPOCHS)...")
    
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"TRAINING {name}")
        print(f"{'='*50}")
        
        callbacks = TrainingUtils.create_callbacks(name.lower())
        
        if USE_GENERATORS:
            # Training dengan generators
            history = model.fit(
                train_gen,
                epochs=EPOCHS,
                validation_data=val_gen,
                callbacks=callbacks,
                verbose=1
            )
        else:
            # Training dengan data langsung
            history = model.fit(
                X_train, y_train,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=1
            )
        
        history_dict[name] = history
        print(f"✅ {name} training selesai!")
    
    # Evaluasi models
    print(f"\n📊 EVALUASI MODEL...")
    
    if not USE_GENERATORS and X_test is not None:
        results = {}
        for name, model in models.items():
            print(f"\nEvaluating {name}...")
            test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
            results[name] = {
                'test_accuracy': test_accuracy,
                'test_loss': test_loss
            }
            print(f"🎯 {name} - Test Accuracy: {test_accuracy:.4f}")
        
        # Visualisasi results
        TrainingUtils.plot_training_history(history_dict)
        
        # Evaluasi detail untuk model terakhir
        print(f"\n🔍 DETAILED EVALUATION UNTUK {list(models.keys())[-1]}:")
        TrainingUtils.evaluate_model(
            list(models.values())[-1], 
            X_test, y_test, 
            class_names
        )
        
        # Plot perbandingan model
        TrainingUtils.plot_model_comparison(results)
    
    print(f"\n✅ SEMUA PROSES SELESAI!")
    print(f"💾 Model disimpan sebagai: saves/best_[model_name].h5")
    print(f"📊 Log training disimpan sebagai: saves/training_[model_name].csv")

if __name__ == "__main__":
    main()