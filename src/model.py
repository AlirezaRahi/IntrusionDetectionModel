# 4. src/model.py
import tensorflow as tf
from keras import layers, Model
from keras.regularizers import l2

class IntrusionDetectionModel:
    def __init__(self, input_shape, num_classes, sequence_length):
        """
        CNN + LSTM model for intrusion detection
        Args:
            input_shape: Number of features
            num_classes: Number of classes
            sequence_length: Time sequence length
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.model = None
        
    def build_model(self):
        """
        Build CNN + LSTM architecture
        """
        print("Building CNN + LSTM model...")
        
        # Input
        inputs = layers.Input(shape=(self.sequence_length, self.input_shape))
        
        # CNN blocks
        x = layers.Conv1D(filters=64, kernel_size=3, padding='same', 
                        kernel_regularizer=l2(0.001))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Conv1D(filters=128, kernel_size=3, padding='same',
                        kernel_regularizer=l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Conv1D(filters=256, kernel_size=3, padding='same',
                        kernel_regularizer=l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(0.3)(x)
        
        # Bidirectional LSTM layers
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True,
                                            kernel_regularizer=l2(0.001)))(x)
        x = layers.Bidirectional(layers.LSTM(64, kernel_regularizer=l2(0.001)))(x)
        x = layers.Dropout(0.4)(x)
        
        # Dense layers
        x = layers.Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = layers.Dropout(0.4)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model.summary()
        return self.model
    
    def get_callbacks(self, checkpoint_path):
        """
        Define callbacks for training
        """
        callbacks = [
            # Save best model
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            # Reduce learning rate
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            ),
            # Early stopping
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            )
        ]
        return callbacks