import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def create_improved_audio_cnn_model(input_shape, num_classes=10):
    """
    Create an improved CNN model for audio genre classification
    
    Args:
        input_shape (tuple): Shape of input spectrograms
        num_classes (int): Number of music genres to classify
    
    Returns:
        tf.keras.Model: Compiled CNN model
    """
    model = models.Sequential([
        # First Convolutional Block - More regularization
        layers.Conv2D(32, (3, 3), activation='relu', 
                      kernel_regularizer=regularizers.l2(0.001),
                      input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.SpatialDropout2D(0.2),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu',
                      kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.SpatialDropout2D(0.3),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu',
                      kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.SpatialDropout2D(0.4),
        
        # Global Average Pooling for better feature representation
        layers.GlobalAveragePooling2D(),
        
        # Fully Connected Layers with Dropout
        layers.Dense(256, activation='relu', 
                     kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu', 
                     kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.5),
        
        # Output Layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Advanced optimizer with learning rate decay
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=100,
            decay_rate=0.9
        )
    )
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model