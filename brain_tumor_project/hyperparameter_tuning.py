# hyperparameter_tuning.py
import tensorflow as tf
import keras_tuner as kt
from dataset_loader import BrainTumorDataLoader
from model_resnet50 import create_resnet50_model

def resnet_hyperparameter_tuning():
    """Hyperparameter tuning untuk ResNet50"""
    
    def build_model(hp):
        # Hyperparameters to tune
        trainable_layers = hp.Int('trainable_layers', 20, 60, step=10)
        learning_rate = hp.Choice('learning_rate', [1e-4, 5e-5, 1e-5])
        dropout_rate = hp.Float('dropout_rate', 0.3, 0.7, step=0.1)
        dense_units = hp.Int('dense_units', 256, 1024, step=128)
        
        model = create_resnet50_model(
            trainable_layers=trainable_layers
        )
        
        # Custom compilation dengan hyperparameters
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    # Setup tuner
    tuner = kt.BayesianOptimization(
        build_model,
        objective='val_accuracy',
        max_trials=10,
        directory='hyperparameter_tuning',
        project_name='resnet50_optimization'
    )
    
    # Load data
    data_loader = BrainTumorDataLoader()
    train_gen, val_gen = data_loader.get_data_generators(batch_size=32)
    
    # Search
    print("🎯 Starting Hyperparameter Search...")
    tuner.search(
        train_gen,
        epochs=10,
        validation_data=val_gen,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5),
            tf.keras.callbacks.ReduceLROnPlateau(patience=3)
        ]
    )
    
    # Get best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    print(f"""
    🏆 BEST HYPERPARAMETERS:
    - Trainable Layers: {best_hps.get('trainable_layers')}
    - Learning Rate: {best_hps.get('learning_rate')}
    - Dropout Rate: {best_hps.get('dropout_rate')}
    - Dense Units: {best_hps.get('dense_units')}
    """)
    
    return tuner, best_hps

if __name__ == "__main__":
    tuner, best_hps = resnet_hyperparameter_tuning()