import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from sklearn.manifold import TSNE

# Import functions from your project
from config import *
from data_loader import prepare_dataloaders
from data_extractor import *
import matplotlib.pyplot as plt
import seaborn as sns



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. LOAD DATA
    print("\n1. Preparing data...")
    dataloaders = prepare_dataloaders(
        original_dir=ORIGINAL_DATA_DIR, 
        augmented_dir=AUGMENTED_DATA_DIR, 
        batch_size=32, 
        use_augmented=True # Set to False if you do not want to use augmented images for testing
    )
    # Get Train set (as dictionary) and Test set (for evaluation)
    train_loader = dataloaders['train']
    test_loader = dataloaders['test']
    
    # 2. LOAD TRAINED MODEL
    print("\n2. Loading model weights...")
    model = RiceFeatureExtractor(embedding_dim=128, use_pretrained=False).to(device)
    
    # Read .pth file and load into model
    model.load_state_dict(torch.load("rice_feature_extractor.pth", map_location=device))
    print("Successfully loaded rice_feature_extractor.pth!")
    
    # 3. EXTRACT FEATURES
    print("\n3. Extracting vectors for Train set (reference dictionary)...")
    train_features, train_labels = extract_features(model, train_loader, device)
    
    print("\nExtracting vectors for Test set (for evaluation)...")
    test_features, test_labels = extract_features(model, test_loader, device)
    
    # 4. EVALUATE USING K-NEAREST NEIGHBORS (KNN)
    print("\n4. EVALUATING ACCURACY WITH KNN...")
    # Initialize KNN algorithm, finding the 5 closest images
    knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
    
    # Fit KNN with the Train dictionary
    knn.fit(train_features, train_labels)
    
    # Predict on the Test set
    predictions = knn.predict(test_features)
    
    # Calculate results
    acc = accuracy_score(test_labels, predictions)
    print(f"\nTEST SET ACCURACY: {acc * 100:.2f}%")
    print("\nDetails per rice variety:")
    print(classification_report(test_labels, predictions, target_names=RICE_VARIETIES))    
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_results = tsne.fit_transform(test_features)
    
    # 2. PREPARE COLORS: Create a distinct color palette for each rice variety
    num_classes = len(RICE_VARIETIES)
    palette = sns.color_palette("husl", num_classes) 
    
    # 3. PLOT CHART: Scatter points on the 2D plane
    plt.figure(figsize=(10, 8)) # Adjust frame size
    sns.scatterplot(
        x=tsne_results[:, 0], 
        y=tsne_results[:, 1],
        hue=test_labels,
        palette=palette,
        legend="full",
        alpha=0.7 # Slightly transparent to clearly see overlapping points
    )
    
    # Decorate the plot for better visibility
    plt.title("t-SNE Clustering of Rice Varieties")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # Move legend outside to avoid overlapping with the plot
    plt.tight_layout() # Adjust layout so the legend is not cropped
    plt.show()

# Use a palette with more colors, e.g., tab20 (max 20 colors) or husl