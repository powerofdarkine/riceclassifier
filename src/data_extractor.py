import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from tqdm import tqdm # Progress bar library
import numpy as np


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
# MAIN TRAINING LOOP
# =====================================================================
def train_model(model, train_loader, num_epochs=10, learning_rate=1e-4, device='cpu'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"\nSTARTING TRAINING ON: {str(device).upper()}")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        model.train() # Switch to train mode
        running_loss = 0.0
        
        # Progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
        
        for images, labels in progress_bar:
            # 1. Move data to device
            images = images.to(device)
            labels = labels.to(device)
            
            # 2. Clear previous gradients
            optimizer.zero_grad()
            
            # 3. FORWARD PASS: Get feature vectors
            embeddings = model(images, return_projection=True)
            
            # 4. CALCULATE LOSS
            loss = triplet_loss(embeddings, labels, margin=1.0)
            
            # Only update weights if loss > 0
            if loss.item() > 0:
                # 5. BACKWARD PASS: Backpropagation
                loss.backward()
                
                # 6. OPTIMIZER: Update weights
                optimizer.step()
            
            # Update progress bar metrics
            running_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        epoch_loss = running_loss / len(train_loader)
        print(f"End of Epoch {epoch+1} | Average Loss: {epoch_loss:.4f}\n")
        
    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Prepare DataLoader (Remember to increase batch_size to 16 or 32 to ensure variety in a batch)
    print("Loading data from DataLoader...")
    dataloaders = prepare_dataloaders(
        original_dir=ORIGINAL_DATA_DIR, 
        augmented_dir=AUGMENTED_DATA_DIR, 
        batch_size=32, 
        use_augmented=True
    )
    train_loader = dataloaders['train']
    
    # 2. Initialize Model
    model = RiceFeatureExtractor(embedding_dim=128, use_pretrained=True)
    
    # 3. Start training
    # Set the number of epochs according to your preference
    trained_model = train_model(
        model=model, 
        train_loader=train_loader, 
        num_epochs=10, 
        learning_rate=1e-4, 
        device=device
    )
    
    # 4. Save the Model
    save_path = "rice_feature_extractor.pth"
    torch.save(trained_model.state_dict(), save_path)
    print(f"Model successfully saved to: {save_path}")
    
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