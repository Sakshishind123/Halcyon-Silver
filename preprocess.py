# import os
# import cv2
# import numpy as np
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# import pandas as pd
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# # print(cv2.__version__)
# # Paths to datasets
# # Paths to datasets using forward slashes
# TRAIN_CSV = 'C:/Users/saksh/OneDrive/Desktop/COEP/datasets/train'
# VALIDATION_CSV = 'C:/Users/saksh/OneDrive/Desktop/COEP/datasets/val/written_name_validation_v2.csv'
# TEST_CSV = 'C:/Users/saksh/OneDrive/Desktop/COEP/datasets/test/written_name_test_v2.csv'


# # IMG_DIR = 'path_to_images'  # Update with the path to the image directory

# # Define character set (modify based on your dataset)
# characters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

# # Mapping characters to integers
# char_to_num = {char: idx + 1 for idx, char in enumerate(characters)}
# num_to_char = {v: k for k, v in char_to_num.items()}

# # Maximum length of a label (depends on your dataset)
# MAX_LABEL_LEN = 32

# def preprocess_image(image_path, img_size=(128, 32)):
#     """Load and preprocess image."""
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     img = cv2.resize(img, img_size)
#     img = img.astype(np.float32) / 255.0  # Normalize
#     img = np.expand_dims(img, axis=-1)
#     return img

# def encode_label(label):
#     """Convert label to a list of integers."""
#     return [char_to_num[char] for char in label if char in char_to_num]

# def load_data(csv_file, img_dir):
#     """Load data from CSV and prepare images and labels."""
#     data = pd.read_csv(csv_file)
#     images = []
#     labels = []
    
#     for idx, row in data.iterrows():
#         image_path = os.path.join(img_dir, row['FILENAME'])
#         label = row['IDENTITY']
        
#         # Process image and label
#         if os.path.exists(image_path):
#             image = preprocess_image(image_path)
#             label = encode_label(label)
            
#             images.append(image)
#             labels.append(label)
    
#     images = np.array(images)
#     labels = pad_sequences(labels, maxlen=MAX_LABEL_LEN, padding='post', value=0)
    
#     return images, labels

# # Example usage:
# # train_images, train_labels = load_data(TRAIN_CSV, IMG_DIR)











import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# Paths to datasets (update with your paths)
TRAIN_CSV = 'C:/Users/saksh/OneDrive/Desktop/COEP/datasets/train/written_name_train_v2.csv'
VALIDATION_CSV = 'C:/Users/saksh/OneDrive/Desktop/COEP/datasets/val/written_name_validation_v2.csv'
TEST_CSV = 'C:/Users/saksh/OneDrive/Desktop/COEP/datasets/test/written_name_test_v2.csv'

# Paths to image directories
TRAIN_IMG_DIR = r'C:\Users\saksh\OneDrive\Desktop\COEP\datasets\train\train_v2'
VALIDATION_IMG_DIR = r'C:\Users\saksh\OneDrive\Desktop\COEP\datasets\val\validation_v2'
TEST_IMG_DIR=r'C:\Users\saksh\OneDrive\Desktop\COEP\datasets\test\test_v2'
# Define character set (modify based on your dataset)
characters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

# Mapping characters to integers
char_to_num = {char: idx + 1 for idx, char in enumerate(characters)}
num_to_char = {v: k for k, v in char_to_num.items()}

# Maximum length of a label (depends on your dataset)
MAX_LABEL_LEN = 32

def preprocess_image(image_path, img_size=(128, 32)):
    """Load and preprocess image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, img_size)
    img = img.astype(np.float32) / 255.0  # Normalize
    img = np.expand_dims(img, axis=-1)
    return img
input_shape = (128, 32, 1)
def encode_label(label):
    """Convert label to a list of integers."""
    return [char_to_num[char] for char in label if char in char_to_num]

def load_data(csv_file, img_dir):
    """Load data from CSV and prepare images and labels."""
    data = pd.read_csv(csv_file)
    images = []
    labels = []
    
    for idx, row in data.iterrows():
        image_path = os.path.join(img_dir, row['FILENAME'])
        label = row['IDENTITY']
        
        # Process image and label
        if os.path.exists(image_path):
            image = preprocess_image(image_path)
            label = encode_label(label)
            
            images.append(image)
            labels.append(label)
    
    images = np.array(images)
    labels = pad_sequences(labels, maxlen=MAX_LABEL_LEN, padding='post', value=0)
    
    return images, labels

# Example usage:
train_images, train_labels = load_data(TRAIN_CSV, TRAIN_IMG_DIR)
validation_images, validation_labels = load_data(VALIDATION_CSV, VALIDATION_IMG_DIR)
test_images,test_labels=load_data(TEST_CSV,TEST_IMG_DIR)

print(f'Training data loaded: {len(train_images)} images, {len(train_labels)} labels')
print(f'Validation data loaded: {len(validation_images)} images, {len(validation_labels)} labels')
print(f'Test data loaded: {len(test_images)} images, {len(test_labels)} labels')