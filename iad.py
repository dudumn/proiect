"""
Human Segmentation AI - Train and Segment People in Videos
This implementation uses PyTorch and a U-Net architecture for human segmentation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
try:
    import imageio
except ImportError as e:
    raise ImportError(
        "The 'imageio' module is required but was not found; install it with: pip install imageio"
    ) from e
import numpy as np
from PIL import Image
import os
from pathlib import Path
from tqdm import tqdm

# ============================================================================
# U-Net Model Architecture
# ============================================================================

class UNet(nn.Module):
    """U-Net architecture for semantic segmentation"""
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder (downsampling)
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder (upsampling)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        # Output
        self.out = nn.Conv2d(64, out_channels, 1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder with skip connections
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return torch.sigmoid(self.out(d1))


# ============================================================================
# Dataset Class
# ============================================================================

class HumanSegmentationDataset(Dataset):
    """Dataset loader for human segmentation (images + masks)"""
    def __init__(self, image_dir, mask_dir, transform=None, img_size=(256, 256)):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.img_size = img_size
        
        # Get all image files
        self.images = sorted([f for f in os.listdir(image_dir) 
                            if f.endswith(('.jpg', '.jpeg', '.png'))])
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_dir / self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Load mask (try different possible mask names)
        mask_name = self.images[idx].replace('.jpg', '.png').replace('.jpeg', '.png')
        mask_path = self.mask_dir / mask_name
        
        if not mask_path.exists():
            # Try with _mask suffix
            base_name = Path(self.images[idx]).stem
            mask_path = self.mask_dir / f"{base_name}_mask.png"
        
        mask = Image.open(mask_path).convert('L')
        
        # Resize
        image = image.resize(self.img_size, Image.BILINEAR)
        mask = mask.resize(self.img_size, Image.NEAREST)
        
        # Convert to tensor
        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)
        
        # Normalize image
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])(image)
        
        return image, mask


# ============================================================================
# Training Class
# ============================================================================

class HumanSegmentationTrainer:
    """Training pipeline for human segmentation model"""
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
    def dice_loss(self, pred, target):
        """Dice loss for better segmentation"""
        smooth = 1e-5
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        
        for images, masks in tqdm(dataloader, desc='Training'):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            
            # Combined loss (BCE + Dice)
            bce_loss = self.criterion(outputs, masks)
            dice = self.dice_loss(outputs, masks)
            loss = bce_loss + dice
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for images, masks in dataloader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def train(self, train_loader, val_loader, epochs=50, save_path='best_model.pth'):
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}')
            print(f'  Val Loss: {val_loss:.4f}')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
                print(f'  Model saved! (Val Loss: {val_loss:.4f})')
            print()


# ============================================================================
# Video Segmentation Class
# ============================================================================

class VideoSegmenter:
    """Segment humans in videos using trained model"""
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = UNet(in_channels=3, out_channels=1).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def segment_frame(self, frame):
        """Segment a single frame (expects an RGB numpy array)"""
        h, w = frame.shape[:2]
        
        # Prepare frame (assume frame is RGB from imageio)
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        frame_pil = Image.fromarray(frame)
        frame_resized = frame_pil.resize((256, 256), Image.BILINEAR)
        
        # Transform and predict
        input_tensor = self.transform(frame_resized).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            mask = self.model(input_tensor)
        
        # Post-process mask
        mask = mask.squeeze().cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8) * 255
        mask_pil = Image.fromarray(mask)
        mask_pil = mask_pil.resize((w, h), Image.NEAREST)
        mask = np.array(mask_pil)
        return mask

    def segment_video(self, input_path, output_path, overlay=True):
        """Segment all frames in a video"""
        # Use imageio for video I/O (frames are RGB)
        reader = imageio.get_reader(input_path)
        meta = reader.get_meta_data()
        fps = meta.get('fps', 25)
        try:
            total_frames = reader.count_frames()
        except Exception:
            total_frames = None
        
        # Determine frame size from the first frame
        first_frame = reader.get_data(0)
        height, width = first_frame.shape[:2]
        
        # Write using imageio (libx264 if available)
        writer = imageio.get_writer(output_path, fps=fps, codec='libx264')
        
        print(f'Processing video: {total_frames if total_frames is not None else "unknown"} frames')
        
        for frame in tqdm(reader, desc='Segmenting'):
            # frame is RGB numpy array
            mask = self.segment_frame(frame)
            
            if overlay:
                overlay_frame = frame.copy()
                overlay_frame[mask > 0] = [0, 255, 0]  # Green overlay (RGB)
                result = (frame.astype(np.float32) * 0.7 + overlay_frame.astype(np.float32) * 0.3).astype(np.uint8)
            else:
                mask_color = np.stack([mask] * 3, axis=2)
                result = np.hstack([frame, mask_color])
            
            writer.append_data(result)
        reader.close()
        writer.close()
        print(f'Video saved to: {output_path}')
        print(f'Video saved to: {output_path}')


# ============================================================================
# Main Usage Example
# ============================================================================

if __name__ == '__main__':
    # Configuration
    IMAGE_DIR = 'dataset/images'
    MASK_DIR = 'dataset/masks'
    MODEL_PATH = 'human_segmentation_model.pth'
    
    # ========== TRAINING ==========
    print("=" * 60)
    print("TRAINING MODE")
    print("=" * 60)
    
    # Create dataset
    dataset = HumanSegmentationDataset(IMAGE_DIR, MASK_DIR, img_size=(256, 256))
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    # Initialize model and trainer
    model = UNet(in_channels=3, out_channels=1)
    trainer = HumanSegmentationTrainer(model)
    
    # Train model
    trainer.train(train_loader, val_loader, epochs=50, save_path=MODEL_PATH)
    
    # ========== VIDEO SEGMENTATION ==========
    print("\n" + "=" * 60)
    print("VIDEO SEGMENTATION MODE")
    print("=" * 60)
    
    # Segment video
    segmenter = VideoSegmenter(MODEL_PATH)
    segmenter.segment_video(
        input_path='input_video.mp4',
        output_path='output_segmented.mp4',
        overlay=True  # Set False for side-by-side view
    )
    
    print("\nDone!")