import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from tqdm import tqdm # Progress bar library
import numpy as np
import matplotlib.pyplot as plt



# Import from your files (Keep unchanged)
from config import *
from data_loader import prepare_dataloaders

class RiceFeatureExtractor(nn.Module):
    def __init__(self, embedding_dim=128, use_pretrained=True):
        super(RiceFeatureExtractor, self).__init__()
        # 1. Base Encoder: Use ResNet18
        resnet = models.resnet18(pretrained=use_pretrained)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        # 2. Projection Head
        self.projection_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x, return_projection=True):
        h = self.encoder(x)
        h = torch.flatten(h, 1) 
        h_norm = F.normalize(h, p=2, dim=1) 
        
        if not return_projection:
            return h_norm 
            
        z = self.projection_head(h_norm)
        z_norm = F.normalize(z, p=2, dim=1)
        return z_norm

# =====================================================================
# LOSS Function - Triplet Loss
# =====================================================================
def triplet_loss(embeddings, labels, margin=1.0):
    """
    Automatically scan within the batch:
    - Find 2 images of the same rice variety to pull them closer.
    - Find 1 image of a different rice variety to push them apart.
    """
    # Calculate the distance matrix between all images in the batch
    distance_matrix = torch.cdist(embeddings, embeddings, p=2)
    
    loss = 0.0
    valid_triplets = 0
    B = embeddings.size(0)
    
    for i in range(B):
        # Get indices of images of the SAME class (Positive) but not itself
        pos_indices = (labels == labels[i]).nonzero(as_tuple=True)[0]
        pos_indices = pos_indices[pos_indices != i]
        
        # Get indices of images of DIFFERENT classes (Negative)
        neg_indices = (labels != labels[i]).nonzero(as_tuple=True)[0]
        
        # If the batch has enough images to form a pair (at least 1 positive, 1 negative)
        if len(pos_indices) > 0 and len(neg_indices) > 0:
            # Randomly select 1 positive and 1 negative
            p = pos_indices[torch.randint(0, len(pos_indices), (1,))[0]]
            n = neg_indices[torch.randint(0, len(neg_indices), (1,))[0]]
            
            # Calculate loss: max(0, positive_distance - negative_distance + margin)
            d_ap = distance_matrix[i, p]
            d_an = distance_matrix[i, n]
            
            triplet_loss = F.relu(d_ap - d_an + margin)
            loss += triplet_loss
            valid_triplets += 1
            
    # Return the average loss of valid pairs
    if valid_triplets > 0:
        return loss / valid_triplets
    else:
        # If the batch has no valid pairs (rare), skip it
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)




# =====================================================================
# MAIN TRAINING LOOP (WITH LOSS TRACKING & EARLY STOPPING)
# =====================================================================
def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=1e-4, device='cpu', patience=5):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Early Stopping & Loss Tracking variables
    best_val_loss = float('inf') 
    epochs_no_improve = 0        
    best_model_path = "rice_feature_extractor.pth"
    
    # Lists to store loss values for plotting
    history_train_loss = []
    history_val_loss = []
    
    print(f"\nSTARTING TRAINING ON: {str(device).upper()} (Max {num_epochs} Epochs)")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        # -----------------------------------------
        # 1. TRAINING PHASE
        # -----------------------------------------
        model.train() 
        running_train_loss = 0.0
        train_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            embeddings = model(images, return_projection=True)
            loss = triplet_loss(embeddings, labels, margin=1.0)
            
            if loss.item() > 0:
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item()
                train_batches += 1
                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
                
        epoch_train_loss = running_train_loss / max(1, train_batches)
        history_train_loss.append(epoch_train_loss)
        
        # -----------------------------------------
        # 2. VALIDATION PHASE
        # -----------------------------------------
        model.eval() 
        running_val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                embeddings = model(images, return_projection=True)
                val_loss = triplet_loss(embeddings, labels, margin=1.0)
                
                if val_loss.item() > 0:
                    running_val_loss += val_loss.item()
                    val_batches += 1
                    
        epoch_val_loss = running_val_loss / max(1, val_batches)
        history_val_loss.append(epoch_val_loss)
        
        print(f"Epoch {epoch+1:02d} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
        
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"   -> New best model saved to: {best_model_path}")
        else:
            epochs_no_improve += 1
            print(f"   -> Val Loss did not improve. (Patience {epochs_no_improve}/{patience})")
            
            if epochs_no_improve >= patience:
                print("\nEARLY STOPPING TRIGGERED! Model might be starting to overfit.")
                print(f"Stopped at Epoch {epoch+1}.")
                break 

    print("-" * 60)
    print(f"Training completed! Best Val Loss: {best_val_loss:.4f}")
    
    # Load the best weights back into the model before returning
    model.load_state_dict(torch.load(best_model_path))
    
    # Return model and the tracked loss histories
    return model, history_train_loss, history_val_loss

# =====================================================================
# PLOTTING LEARNING CURVES
# =====================================================================
def plot_learning_curves(train_losses, val_losses, save_path="learning_curves.png"):
    """
    Plots the Training and Validation Loss to check for Overfitting.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', marker='o', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', marker='o', linewidth=2)
    
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Triplet Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\nLearning curves plotted and saved to '{save_path}'")
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading data from DataLoader...")
    dataloaders = prepare_dataloaders(
        original_dir=ORIGINAL_DATA_DIR, 
        augmented_dir=AUGMENTED_DATA_DIR, 
        batch_size=32, 
        use_augmented=True
    )
    
    # Ensure you are getting both train and val loaders
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    
    model = RiceFeatureExtractor(embedding_dim=128, use_pretrained=True)
    
    # Train the model and get the loss histories
    trained_model, train_losses, val_losses = train_model(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader,
        num_epochs=30,  # You can increase this, early stopping will prevent overfitting
        learning_rate=1e-4, 
        device=device,
        patience=10
    )
    
    # Plot the learning curves after training is done
    plot_learning_curves(train_losses, val_losses)
def extract_features(model, dataloader, device):
    """
    Run all images through the model and collect vectors into a Numpy array
    """
    model.eval() # MANDATORY: Set model to evaluation mode (disable Dropout, freeze BatchNorm)
    
    all_features = []
    all_labels = []
    
    with torch.no_grad(): # Disable gradient calculation to save RAM and speed up execution
        for images, labels in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            
            # Note: return_projection=False to extract only the core feature vector from ResNet18
            features = model(images, return_projection=False) 
            
            # Move from GPU to CPU and convert to Numpy array
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
            
    # Concatenate batches into a single array
    return np.vstack(all_features), np.concatenate(all_labels)