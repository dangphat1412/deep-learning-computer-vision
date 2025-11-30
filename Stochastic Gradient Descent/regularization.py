import argparse
import numpy as np
from imutils import paths
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from utilities.preprocessing import SimplePreprocessor
from utilities.datasets import SimpleDatasetLoader


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train SGD classifier on image dataset')
    parser.add_argument('-d', '--dataset', required=True, 
                       help='Path to input dataset')
    parser.add_argument('-e', '--epochs', type=int, default=10,
                       help='Number of training epochs (default: 10)')
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.01,
                       help='Learning rate (default: 0.01)')
    return vars(parser.parse_args())


def load_and_preprocess_data(dataset_path, target_size=(32, 32)):
    """
    Load images from dataset path and preprocess them.
    
    Args:
        dataset_path: Path to dataset directory
        target_size: Target image size (width, height)
    
    Returns:
        data: Flattened image data
        labels: Image labels
        label_encoder: Fitted LabelEncoder
    """
    print('[INFO] Loading images...')
    
    # Get image paths
    image_paths = list(paths.list_images(dataset_path))
    print(f'[INFO] Found {len(image_paths)} images')
    
    # Load and preprocess images
    preprocessor = SimplePreprocessor(*target_size)
    loader = SimpleDatasetLoader(preprocessors=[preprocessor])
    data, labels = loader.load(image_paths, verbose=500)
    
    # Flatten images: (N, H, W, C) -> (N, H*W*C)
    n_samples = data.shape[0]
    n_features = np.prod(data.shape[1:])
    data = data.reshape((n_samples, n_features))
    
    # Print memory usage
    memory_mb = data.nbytes / (1024 ** 2)
    print(f'[INFO] Features matrix: {memory_mb:.1f}MB')
    print(f'[INFO] Shape: {data.shape}')
    
    # Encode labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    print(f'[INFO] Classes: {label_encoder.classes_}')
    
    return data, labels, label_encoder


def train_and_evaluate_model(train_x, train_y, test_x, test_y, 
                             penalty=None, max_iter=10, learning_rate=0.01):
    """
    Train SGD classifier with specified regularization and evaluate.
    
    Args:
        train_x, train_y: Training data and labels
        test_x, test_y: Test data and labels
        penalty: Regularization type (None, 'l1', 'l2')
        max_iter: Number of epochs
        learning_rate: Learning rate
    
    Returns:
        model: Trained model
        accuracy: Test accuracy
    """
    penalty_name = str(penalty).upper() if penalty else 'NONE'
    print(f'\n[INFO] Training model with {penalty_name} penalty')
    
    # Initialize model
    model = SGDClassifier(
        loss='log_loss',
        penalty=penalty if penalty else None,
        max_iter=max_iter,
        learning_rate='constant',
        eta0=learning_rate,
        random_state=42,
        tol=1e-3
    )
    
    # Train model
    model.fit(train_x, train_y)
    
    # Evaluate model
    accuracy = model.score(test_x, test_y)
    print(f'[INFO] {penalty_name} penalty accuracy: {accuracy * 100:.2f}%')
    
    return model, accuracy

def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    
    # Load and preprocess data
    data, labels, label_encoder = load_and_preprocess_data(args['dataset'])
    
    # Split data into train/test sets
    train_x, test_x, train_y, test_y = train_test_split(
        data, labels,
        test_size=0.25,
        random_state=5,
        stratify=labels  # Ensure balanced split
    )
    print(f'\n[INFO] Training samples: {len(train_x)}')
    print(f'[INFO] Testing samples: {len(test_x)}')
    
    # Train models with different regularization methods
    regularization_methods = [None, 'l1', 'l2']
    results = {}
    
    for reg_method in regularization_methods:
        model, accuracy = train_and_evaluate_model(
            train_x, train_y, test_x, test_y,
            penalty=reg_method,
            max_iter=args['epochs'],
            learning_rate=args['learning_rate']
        )
        results[reg_method] = (model, accuracy)
    
    # Find best model
    best_penalty = max(results, key=lambda k: results[k][1])
    best_model, best_accuracy = results[best_penalty]
    
    print(f'\n[INFO] Best model: {str(best_penalty).upper()} penalty')
    print(f'[INFO] Best accuracy: {best_accuracy * 100:.2f}%')
    
    # Detailed classification report for best model
    print('\n[INFO] Classification Report (Best Model):')
    predictions = best_model.predict(test_x)
    print(classification_report(test_y, predictions, 
                               target_names=label_encoder.classes_))


if __name__ == '__main__':
    main()