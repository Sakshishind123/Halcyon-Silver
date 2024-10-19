# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import layers, models
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# # Define image dimensions
# height = 64    # Replace with your actual image height
# width = 64     # Replace with your actual image width
# channels = 3   # Replace with 1 for grayscale images
# num_classes = 10  # Replace with your actual number of classes
# batch_size = 32 
# num_epochs = 10  # You can adjust the number of epochs as needed

# def ctc_loss_lambda_func(args):
#     y_pred, labels, input_length, label_length = args
#     return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)

# def build_crnn(input_shape, num_classes):
#     """Build CRNN model with CTC loss."""
#     inputs = layers.Input(shape=input_shape)
    
#     # CNN Layers
#     x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
#     x = layers.MaxPooling2D(pool_size=(2, 2))(x)
#     x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
#     x = layers.MaxPooling2D(pool_size=(2, 2))(x)

#     # Reshape for RNN layers
#     new_shape = (input_shape[0] // 4, (input_shape[1] // 4) * 64)
#     x = layers.Reshape(target_shape=new_shape)(x)

#     # Bidirectional LSTM layers
#     x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
#     x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    
#     # Dense layer to predict character classes
#     x = layers.Dense(num_classes + 1, activation='softmax')(x)

#     # Define CTC inputs and outputs
#     labels = layers.Input(shape=(None,), dtype='float32', name='labels')
#     input_length = layers.Input(shape=(1,), dtype='int64', name='input_length')
#     label_length = layers.Input(shape=(1,), dtype='int64', name='label_length')

#     # CTC loss layer
#     loss_out = layers.Lambda(ctc_loss_lambda_func, output_shape=(1,), name='ctc')([x, labels, input_length, label_length])

#     # Model definition
#     model = models.Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)
    
#     return model

# # Build and compile the model
# input_shape = (height, width, channels)
# model = build_crnn(input_shape=input_shape, num_classes=num_classes)
# model.compile(optimizer='adam')

# # Set base directory for datasets
# BASE_DIR = r'C:\Users\saksh\OneDrive\Desktop\COEP\datasets'

# # Data Generator (Modify for CTC)
# def ctc_data_generator(data_gen, batch_size):
#     while True:
#         X, y = next(data_gen)
#         input_length = np.ones((batch_size, 1)) * (X.shape[1] // 2)
#         label_length = np.ones((batch_size, 1)) * y.shape[1]
#         yield [X, y, input_length, label_length], np.zeros(batch_size)

# # Load your data using ImageDataGenerator
# train_data_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
#     os.path.join(BASE_DIR, 'train'),
#     target_size=(height, width),
#     batch_size=batch_size,
#     class_mode='categorical'
# )

# # Use custom generator for CTC
# train_data = ctc_data_generator(train_data_gen, batch_size)

# # Train the model
# model.fit(train_data, epochs=num_epochs, steps_per_epoch=len(train_data_gen))

# # Evaluate (CTC requires specific evaluation functions)
# # test_data = ctc_data_generator(...)








import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define image dimensions
height = 64    # Replace with your actual image height
width = 64     # Replace with your actual image width
channels = 3   # Replace with 1 for grayscale images
batch_size = 32   # Set the batch size
num_epochs = 2   # Set the number of epochs

def build_crnn(input_shape, num_classes):
    """Build CRNN model with CTC loss."""
    inputs = layers.Input(shape=input_shape)
    
    # CNN Layers
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Reshape for RNN layers
    new_shape = (input_shape[0] // 4, (input_shape[1] // 4) * 64)
    x = layers.Reshape(target_shape=new_shape)(x)

    # Bidirectional LSTM layers
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    
    # Apply Global Average Pooling to reduce time dimension
    x = layers.GlobalAveragePooling1D()(x)

    # Dense layer to predict character classes
    x = layers.Dense(num_classes, activation='softmax')(x)

    # Model definition
    model = models.Model(inputs, x)
    
    return model

# Parameters
num_classes = 10  # Replace with the actual number of classes

# Build and compile the model
input_shape = (height, width, channels)
model = build_crnn(input_shape=input_shape, num_classes=num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Set base directory for datasets
BASE_DIR = r'C:\Users\saksh\OneDrive\Desktop\COEP\datasets'  # Adjust the base directory as needed

# Ensure directories exist
if not os.path.exists(os.path.join(BASE_DIR, 'train')):
    print("Train directory does not exist.")
if not os.path.exists(os.path.join(BASE_DIR, 'val')):
    print("Validation directory does not exist.")
if not os.path.exists(os.path.join(BASE_DIR, 'test')):
    print("Test directory does not exist.")

# Load your data using ImageDataGenerator
train_data_gen = ImageDataGenerator(rescale=1./255)
val_data_gen = ImageDataGenerator(rescale=1./255)
test_data_gen = ImageDataGenerator(rescale=1./255)

# Load your data using ImageDataGenerator
train_data = train_data_gen.flow_from_directory(
    os.path.join(BASE_DIR, 'train'),
    target_size=(height, width),
    batch_size=batch_size,
    class_mode='sparse'  # Use 'sparse' if labels are integers
)

val_data = val_data_gen.flow_from_directory(
    os.path.join(BASE_DIR, 'val'),
    target_size=(height, width),
    batch_size=batch_size,
    class_mode='sparse'  # Use 'sparse'
)

test_data = test_data_gen.flow_from_directory(
    os.path.join(BASE_DIR, 'test'),
    target_size=(height, width),
    batch_size=batch_size,
    class_mode='sparse',  # Use 'sparse'
    shuffle=False
)

# Compile the model
model.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics=['accuracy'])  # Update loss function

# Fit the model
model.fit(train_data, 
          epochs=num_epochs, 
          steps_per_epoch=len(train_data), 
          validation_data=val_data, 
          validation_steps=len(val_data))

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_data)
print(f'Test loss: {test_loss}, Test accuracy: {test_accuracy}')
