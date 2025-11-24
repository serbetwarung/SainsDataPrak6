# model_vit.py
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

def create_vit_model(
    input_shape=(224, 224, 3),
    patch_size=16,
    projection_dim=64,
    num_heads=4,
    transformer_layers=8,
    num_classes=4
):
    """
    Membuat Vision Transformer model yang disederhanakan
    """
    # Input layer
    inputs = layers.Input(shape=input_shape)
    
    # Patch creation
    patches = layers.Conv2D(
        projection_dim, 
        kernel_size=patch_size, 
        strides=patch_size, 
        padding='valid'
    )(inputs)
    patches = layers.Reshape((-1, projection_dim))(patches)
    
    # Position embedding
    num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
    positions = tf.range(start=0, limit=num_patches, delta=1)
    position_embedding = layers.Embedding(
        input_dim=num_patches, output_dim=projection_dim
    )(positions)
    
    # Add position embedding to patches
    encoded_patches = patches + position_embedding
    
    # Transformer blocks
    for _ in range(transformer_layers):
        # Layer normalization 1
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=projection_dim//num_heads, 
            dropout=0.1
        )(x1, x1)
        
        # Skip connection 1
        x2 = layers.Add()([attention_output, encoded_patches])
        
        # Layer normalization 2
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        
        # MLP
        x3 = layers.Dense(projection_dim * 2, activation=tf.nn.gelu)(x3)
        x3 = layers.Dropout(0.1)(x3)
        x3 = layers.Dense(projection_dim, activation=tf.nn.gelu)(x3)
        x3 = layers.Dropout(0.1)(x3)
        
        # Skip connection 2
        encoded_patches = layers.Add()([x3, x2])
    
    # Classification head
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.GlobalAveragePooling1D()(representation)
    representation = layers.Dropout(0.5)(representation)
    
    # MLP Head
    features = layers.Dense(256, activation=tf.nn.gelu)(representation)
    features = layers.Dropout(0.5)(features)
    outputs = layers.Dense(num_classes, activation="softmax")(features)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def compile_vit_model(model, learning_rate=0.0001):
    """Compile ViT model"""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def create_enhanced_vit_model(
    input_shape=(224, 224, 3),
    patch_size=16,
    projection_dim=128,  # Increased projection
    num_heads=8,         # Increased heads
    transformer_layers=12, # Increased layers
    mlp_ratio=4,
    num_classes=4
):
    """
    Enhanced Vision Transformer dengan konfigurasi lebih besar
    """
    # Input
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Augmentasi built-in
    x = tf.keras.layers.RandomRotation(0.1)(inputs)
    x = tf.keras.layers.RandomZoom(0.1)(x)
    
    # Patch embedding dengan conv lebih dalam
    x = tf.keras.layers.Conv2D(projection_dim, kernel_size=patch_size, 
                              strides=patch_size, padding='valid')(x)
    x = tf.keras.layers.Reshape((-1, projection_dim))(x)
    
    # Position embedding
    num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
    positions = tf.range(start=0, limit=num_patches, delta=1)
    position_embedding = tf.keras.layers.Embedding(
        input_dim=num_patches, output_dim=projection_dim
    )(positions)
    
    # Add position embedding
    x = x + position_embedding
    x = tf.keras.layers.Dropout(0.1)(x)
    
    # Transformer blocks
    for _ in range(transformer_layers):
        # Skip connection 1
        x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Multi-head attention
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=projection_dim // num_heads,
            dropout=0.1
        )(x1, x1)
        
        x2 = tf.keras.layers.Add()([attention_output, x])
        
        # Skip connection 2
        x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)
        
        # MLP dengan lebih banyak capacity
        mlp_units = projection_dim * mlp_ratio
        x3 = tf.keras.layers.Dense(mlp_units, activation=tf.nn.gelu)(x3)
        x3 = tf.keras.layers.Dropout(0.1)(x3)
        x3 = tf.keras.layers.Dense(projection_dim, activation=tf.nn.gelu)(x3)
        x3 = tf.keras.layers.Dropout(0.1)(x3)
        
        x = tf.keras.layers.Add()([x3, x2])
    
    # Enhanced classification head
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    x = tf.keras.layers.Dense(512, activation=tf.nn.gelu)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    x = tf.keras.layers.Dense(256, activation=tf.nn.gelu)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model

def create_vit_variants():
    """Membuat beberapa variant ViT untuk eksperimen"""
    variants = {}
    
    # Variant 1: Small ViT
    variants['small'] = create_vit_model(
        projection_dim=64,
        num_heads=4,
        transformer_layers=6
    )
    
    # Variant 2: Medium ViT (default)
    variants['medium'] = create_vit_model(
        projection_dim=64,
        num_heads=8,
        transformer_layers=8
    )
    
    # Variant 3: Large ViT
    variants['large'] = create_vit_model(
        projection_dim=128,
        num_heads=8,
        transformer_layers=12
    )
    
    # Variant 4: Enhanced ViT
    variants['enhanced'] = create_enhanced_vit_model()
    
    return variants

# Contoh penggunaan
if __name__ == "__main__":
    print("🧪 Testing Vision Transformer Models...")
    
    # Test basic model
    print("\n1. Basic ViT Model:")
    model_basic = create_vit_model()
    model_basic = compile_vit_model(model_basic)
    model_basic.summary()
    
    # Test enhanced model
    print("\n2. Enhanced ViT Model:")
    model_enhanced = create_enhanced_vit_model()
    model_enhanced = compile_vit_model(model_enhanced, learning_rate=0.00005)
    model_enhanced.summary()
    
    # Test variants
    print("\n3. ViT Variants:")
    variants = create_vit_variants()
    for name, model in variants.items():
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        trainable_params = sum([w.shape.num_elements() for w in model.trainable_weights])
        print(f"   {name}: {trainable_params:,} trainable parameters")
    
    print("\n✅ All Vision Transformer models created successfully!")