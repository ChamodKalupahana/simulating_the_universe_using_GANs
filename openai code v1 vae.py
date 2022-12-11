import tensorflow

# Create the VAE model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Input, Conv2DTranspose, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Encoder
input_shape = (256, 256, 3)
inputs = Input(shape=input_shape, name='encoder_input')
x = Conv2D(32, 3, activation='relu', strides=2, padding='same')(inputs)
x = Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)
x = Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)
x = Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)
# shape info needed to build decoder Model
shape = K.int_shape(x)

# Build the latent vector
x = Flatten()(x)
latent_dim = 32
latent = Dense(latent_dim, name='latent_vector')(x)

# Instantiate encoder Model
encoder = Model(inputs, latent, name='encoder')
encoder.summary()

# Decoder
latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
# Use the shape (shape[1], shape[2], shape[3]) that was earlier stored
x = Dense(shape[1]*shape[2]*shape[3])(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)

x = Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)
x = Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)
x = Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same')(x)
outputs = Conv2DTranspose(3, 3, activation='sigmoid', padding='same')(x)

# Instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')