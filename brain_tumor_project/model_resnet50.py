# model_resnet50.py
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models

def create_resnet50_model(input_shape=(224, 224, 3), num_classes=4, trainable_layers=30):
    """
    Membuat ResNet50 model dengan transfer learning yang lebih optimal
    """
    try:
        # Load pre-trained ResNet50
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        print(f"✅ ResNet50 base model loaded. Total layers: {len(base_model.layers)}")
        
        # Freeze awal: hanya unfreeze sebagian layer
        base_model.trainable = False
        
        # Unfreeze layer tertentu untuk fine-tuning
        if trainable_layers > 0:
            # Unfreeze layer terakhir
            for layer in base_model.layers[-trainable_layers:]:
                layer.trainable = True
            
            print(f"✅ Unfrozen {trainable_layers} layers untuk fine-tuning")
        
        # Build model dengan architecture yang lebih robust
        inputs = tf.keras.Input(shape=input_shape)
        
        # Preprocessing ResNet50
        x = tf.keras.applications.resnet50.preprocess_input(inputs)
        
        # Base model
        x = base_model(x, training=False)  # Important: training=False untuk frozen layers
        
        # Global pooling dengan multiple options
        x1 = layers.GlobalAveragePooling2D()(x)
        x2 = layers.GlobalMaxPooling2D()(x)
        x = layers.Concatenate()([x1, x2])
        
        # Batch normalization
        x = layers.BatchNormalization()(x)
        
        # Dense layers dengan regularization
        x = layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        
        return model
        
    except Exception as e:
        print(f"❌ Error creating ResNet50 model: {e}")
        return None

def compile_resnet50_model(model, learning_rate=0.0001):
    """Compile ResNet50 model dengan optimizer yang lebih baik"""
    if model is None:
        print("❌ Model is None, cannot compile")
        return None
        
    # Optimizer dengan gradient clipping untuk stability
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✅ ResNet50 model compiled with learning rate: {learning_rate}")
    return model

def create_enhanced_resnet50(input_shape=(224, 224, 3), num_classes=4):
    """
    Enhanced ResNet50 dengan progressive unfreezing
    """
    # Load pre-trained ResNet50
    base_model = tf.keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Progressive unfreezing strategy
    base_model.trainable = False
    
    # Stage 1: Unfreeze last 40 layers
    for layer in base_model.layers[-40:]:
        layer.trainable = True
    
    print(f"✅ Enhanced ResNet50: Unfrozen 40 layers untuk fine-tuning")
    
    # Enhanced architecture
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.applications.resnet50.preprocess_input(inputs)
    x = base_model(x, training=False)
    
    # Multiple pooling strategies
    x1 = tf.keras.layers.GlobalAveragePooling2D()(x)
    x2 = tf.keras.layers.GlobalMaxPooling2D()(x)
    x3 = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Concatenate()([x1, x2, x3])
    
    # Enhanced classifier dengan regularisasi
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    x = tf.keras.layers.Dense(512, activation='relu',
                             kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    x = tf.keras.layers.Dense(256, activation='relu',
                             kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model

def create_resnet50_variants():
    """Membuat beberapa variant ResNet50 untuk eksperimen"""
    
    variants = {}
    
    # Variant 1: Fully frozen (feature extraction)
    base_model1 = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model1.trainable = False
    
    model1 = models.Sequential([
        base_model1,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(4, activation='softmax')
    ])
    variants['frozen'] = model1
    
    # Variant 2: Partial fine-tuning
    base_model2 = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model2.trainable = False
    
    # Unfreeze last 30 layers
    for layer in base_model2.layers[-30:]:
        layer.trainable = True
    
    model2 = models.Sequential([
        base_model2,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(4, activation='softmax')
    ])
    variants['partial_ft'] = model2
    
    # Variant 3: Dengan data augmentation built-in
    data_augmentation = tf.keras.Sequential([
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomFlip("horizontal"),
    ])
    
    base_model3 = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model3.trainable = False
    
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.resnet50.preprocess_input(x)
    x = base_model3(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(4, activation='softmax')(x)
    
    model3 = tf.keras.Model(inputs, outputs)
    variants['with_aug'] = model3
    
    # Variant 4: Enhanced version
    variants['enhanced'] = create_enhanced_resnet50()
    
    return variants

def get_resnet50_layers_info():
    """Utility function untuk melihat informasi layer ResNet50"""
    base_model = ResNet50(weights='imagenet', include_top=False)
    
    print("\n📋 ResNet50 Layers Information:")
    print(f"Total layers: {len(base_model.layers)}")
    
    # Tampilkan nama layer terakhir
    print("\nLast 20 layers:")
    for i, layer in enumerate(base_model.layers[-20:]):
        print(f"{len(base_model.layers) - 20 + i}: {layer.name} - Trainable: {layer.trainable}")

# Contoh penggunaan
if __name__ == "__main__":
    print("🧪 Testing ResNet50 Models...")
    
    # Test 1: Model dasar
    print("\n1. Testing basic ResNet50 model:")
    model = create_resnet50_model()
    if model:
        model = compile_resnet50_model(model)
        model.summary()
    
    # Test 2: Enhanced model
    print("\n2. Testing enhanced ResNet50 model:")
    model_enhanced = create_enhanced_resnet50()
    if model_enhanced:
        model_enhanced = compile_resnet50_model(model_enhanced, learning_rate=0.00005)
        model_enhanced.summary()
    
    # Test 3: Dapatkan info layer
    print("\n3. Layer information:")
    get_resnet50_layers_info()
    
    # Test 4: Variants
    print("\n4. Creating model variants...")
    variants = create_resnet50_variants()
    for name, variant_model in variants.items():
        variant_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        trainable_params = sum([w.shape.num_elements() for w in variant_model.trainable_weights])
        non_trainable_params = sum([w.shape.num_elements() for w in variant_model.non_trainable_weights])
        print(f"   {name}: {trainable_params:,} trainable, {non_trainable_params:,} non-trainable")
    
    print("\n✅ All ResNet50 models created successfully!")