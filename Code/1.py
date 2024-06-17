import os
import shutil
from sklearn.model_selection import train_test_split

# Define paths
dataset_dir = 'Dataset'
train_dir = 'Train'
val_dir = 'Validation'
test_dir = 'Test'

# Parameters
test_size = 0.1  # 10% for test set
val_size = 0.2   # 20% for validation set

# Create directories for train, validation, and test sets
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Function to split dataset
def split_dataset(dataset_dir, train_dir, val_dir, test_dir, test_size, val_size):
    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_dir):
            os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

            file_paths = [os.path.join(class_dir, fname) for fname in os.listdir(class_dir)]
            labels = [class_name] * len(file_paths)  # Create a list of labels for stratification
            
            train_paths, test_paths, train_labels, test_labels = train_test_split(
                file_paths, labels, test_size=test_size, stratify=labels, random_state=42)
            
            train_paths, val_paths, train_labels, val_labels = train_test_split(
                train_paths, train_labels, test_size=val_size, stratify=train_labels, random_state=42)
            
            for path in train_paths:
                shutil.copy(path, os.path.join(train_dir, class_name))
            for path in val_paths:
                shutil.copy(path, os.path.join(val_dir, class_name))
            for path in test_paths:
                shutil.copy(path, os.path.join(test_dir, class_name))

# Split the dataset
split_dataset(dataset_dir, train_dir, val_dir, test_dir, test_size, val_size)
