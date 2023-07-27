
from keras.layers import Conv2D, UpSampling2D
from keras.models import Sequential
import tensorflow as tf

from .References import References

class DecoderLayers(References):

    def __init__(self):

        self.model = Sequential()
        self.__setModelArch()

    def __setModelArch(self):
        
                # Add a 2D convolutional layer with 256 filters, a 3x3 kernel, 'relu' activation,
        # 'same' padding, and an input shape of (7, 7, 512)
        self.model.add(Conv2D(256, (3, 3), activation='relu', padding='same', input_shape=(7, 7, 512)))
        
        # Add another 2D convolutional layer with 128 filters, a 3x3 kernel, 'relu' activation,
        # and 'same' padding
        self.model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        
        # Add an upsampling layer that doubles the spatial dimensions (2x) of the input
        self.model.add(UpSampling2D((2, 2)))
        
        # Add another 2D convolutional layer with 64 filters, a 3x3 kernel, 'relu' activation,
        # and 'same' padding
        self.model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        
        # Add an upsampling layer that doubles the spatial dimensions (2x) of the input
        self.model.add(UpSampling2D((2, 2)))
        
        # Add another 2D convolutional layer with 32 filters, a 3x3 kernel, 'relu' activation,
        # and 'same' padding
        self.model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        
        # Add an upsampling layer that doubles the spatial dimensions (2x) of the input
        self.model.add(UpSampling2D((2, 2)))
        
        # Add another 2D convolutional layer with 16 filters, a 3x3 kernel, 'relu' activation,
        # and 'same' padding
        self.model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
        
        # Add an upsampling layer that doubles the spatial dimensions (2x) of the input
        self.model.add(UpSampling2D((2, 2)))
        
        # Add another 2D convolutional layer with 2 filters, a 3x3 kernel, 'tanh' activation,
        # and 'same' padding
        self.model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
        
        # Add an upsampling layer that doubles the spatial dimensions (2x) of the input
        self.model.add(UpSampling2D((2, 2)))

        # Compile the model with 'Adam' optimizer, mean squared error (MSE) loss function,
        # and accuracy as a metric
        self.model.compile(optimizer='Adam', loss='mse', metrics=['accuracy'])

    def fit(self, vggfeatures, Y):
        
        # Fit the model to the training data
        self.model.fit(vggfeatures, Y, verbose=1, epochs=5, batch_size=16)

        # Save the trained model
        self.model.save(self.ROOT_DIR+self.SAVE_MODEL)

    def load_model(self):
        
        # Load the trained model
        self.model = tf.keras.models.load_model(self.ROOT_DIR+self.SAVE_MODEL,
                                           custom_objects=None,
                                           compile=True)
        return self.model

