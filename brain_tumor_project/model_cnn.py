# model_cnn.py
import tensorflow as tf
from tensorflow.keras import layers, models

def create_cnn_model(input_shape=(224, 224, 3), num_classes=4):
    """
    Membuat CNN model untuk klasifikasi tumor otak
    """
    model = models.Sequential([
        # First Conv Block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                     input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Conv Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Conv Block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fourth Conv Block
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Classifier
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def compile_cnn_model(model, learning_rate=0.001):
    """Compile CNN model"""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def create_improved_cnn_model(input_shape=(224, 224, 3), num_classes=4):
    """
    Improved CNN model dengan regularisasi untuk atasi overfitting
    """
    model = tf.keras.Sequential([
        # First Conv Block dengan regularisasi
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                              input_shape=input_shape,
                              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.3),  # Increased dropout
        
        # Second Conv Block
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.4),  # Increased dropout
        
        # Third Conv Block
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.4),
        
        # Fourth Conv Block
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.4),
        
        # Global Average Pooling instead of Flatten untuk reduce overfitting
        tf.keras.layers.GlobalAveragePooling2D(),
        
        # Classifier dengan lebih banyak regularisasi
        tf.keras.layers.Dense(512, activation='relu',
                             kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.6),  # High dropout
        
        tf.keras.layers.Dense(256, activation='relu',
                             kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_cnn_model_variants():
    """Membuat beberapa variant CNN untuk eksperimen"""
    variants = {}
    
    # Variant 1: Simple CNN (baseline)
    variants['simple'] = create_cnn_model()
    
    # Variant 2: Improved CNN dengan regularisasi
    variants['improved'] = create_improved_cnn_model()
    
    # Variant 3: Deep CNN dengan lebih banyak layer
    model_deep = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same', input_shape=(224,224,3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2,2),
        
        tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2,2),
        
        tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2,2),
        
        tf.keras.layers.Conv2D(512, (3,3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2,2),
        
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    variants['deep'] = model_deep
    
    return variants

# Contoh penggunaan
if __name__ == "__main__":
    print("🧪 Testing CNN Models...")
    
    # Test basic model
    print("\n1. Basic CNN Model:")
    model_basic = create_cnn_model()
    model_basic = compile_cnn_model(model_basic)
    model_basic.summary()
    
    # Test improved model
    print("\n2. Improved CNN Model:")
    model_improved = create_improved_cnn_model()
    model_improved = compile_cnn_model(model_improved, learning_rate=0.0005)
    model_improved.summary()
    
    # Test variants
    print("\n3. CNN Variants:")
    variants = create_cnn_model_variants()
    for name, model in variants.items():
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        trainable_params = sum([w.shape.num_elements() for w in model.trainable_weights])
        print(f"   {name}: {trainable_params:,} trainable parameters")
    
    print("\n✅ All CNN models created successfully!")