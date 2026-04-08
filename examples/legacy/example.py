#!/usr/bin/env python3
"""
Example script for using the UHG-based anomaly detection system.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from uhg_anomaly_detection_2 import run_anomaly_detection, preprocess_modbus_data

# Create sample data directory
os.makedirs('sample_data', exist_ok=True)

# Generate synthetic Modbus data for demonstration
def generate_synthetic_modbus_data(n_samples=1000, n_features=10, anomaly_ratio=0.05):
    """Generate synthetic Modbus data for demonstration."""
    print("Generating synthetic Modbus data...")
    
    # Generate normal data
    normal_data = np.random.normal(0, 1, size=(int(n_samples * (1 - anomaly_ratio)), n_features))
    normal_labels = np.zeros(int(n_samples * (1 - anomaly_ratio)))
    
    # Generate anomaly data
    anomaly_data = np.random.normal(3, 2, size=(int(n_samples * anomaly_ratio), n_features))
    anomaly_labels = np.ones(int(n_samples * anomaly_ratio))
    
    # Combine data
    data = np.vstack([normal_data, anomaly_data])
    labels = np.hstack([normal_labels, anomaly_labels])
    
    # Create DataFrame
    columns = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(data, columns=columns)
    df['label'] = labels
    
    # Shuffle data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

# Generate and save synthetic data
def save_synthetic_data():
    """Generate and save synthetic Modbus data."""
    # Generate data
    data = generate_synthetic_modbus_data(n_samples=5000, n_features=20, anomaly_ratio=0.05)
    
    # Split data into train, validation, and test sets
    train_size = int(len(data) * 0.7)
    val_size = int(len(data) * 0.15)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size+val_size]
    test_data = data[train_size+val_size:]
    
    # Save data
    train_file = 'sample_data/train_data.csv'
    val_file = 'sample_data/val_data.csv'
    test_file = 'sample_data/test_data.csv'
    
    train_data.to_csv(train_file, index=False)
    val_data.to_csv(val_file, index=False)
    test_data.to_csv(test_file, index=False)
    
    print(f"Train data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    return train_file, val_file, test_file

def main():
    """Main function to demonstrate UHG-based anomaly detection."""
    print("UHG-Based Anomaly Detection Example")
    print("===================================")
    
    # Generate and save synthetic data
    train_file, val_file, test_file = save_synthetic_data()
    
    # Preprocess data
    print("\nPreprocessing data...")
    preprocessed_paths = preprocess_modbus_data(
        train_file=train_file,
        val_file=val_file,
        test_file=test_file,
        data_percentage=1.0  # Use all data for this example
    )
    
    # Run anomaly detection
    print("\nRunning UHG-based anomaly detection...")
    results = run_anomaly_detection(
        train_file=preprocessed_paths['train'],
        val_file=preprocessed_paths['val'],
        test_file=preprocessed_paths['test'],
        k=2,  # Number of nearest neighbors for graph construction
        epochs=50,  # Reduced for demonstration
        hidden_dim=32,
        latent_dim=16
    )
    
    # Print results
    print("\nResults:")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    print(f"AUC: {results['auc']:.4f}")
    print(f"Confusion Matrix:")
    print(f"  True Negatives: {results['confusion_matrix']['tn']}")
    print(f"  False Positives: {results['confusion_matrix']['fp']}")
    print(f"  False Negatives: {results['confusion_matrix']['fn']}")
    print(f"  True Positives: {results['confusion_matrix']['tp']}")
    
    print(f"\nModel saved to: {results['model_path']}")
    print(f"Plot saved to: {results['plot_path']}")
    
    # Display the plot
    print("\nDisplaying results plot...")
    img = plt.imread(results['plot_path'])
    plt.figure(figsize=(15, 5))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main() 