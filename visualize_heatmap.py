import matplotlib.pyplot as plt
import cv2
from PIL import Image
import torch.nn.functional as F
import torch
import numpy as np
from model import Model
from scipy import ndimage


def visualize_feature_grouping(model, img_path, device='cuda', save_dir='heatmaps'):
    """
    Visualize the heatmaps of 64 categories for the input image.
    """
    model.eval()
    model.to(device)

    # 1. Preprocess the input image
    transform_config = model.get_config()
    mean = torch.tensor(transform_config['mean']).view(1, 3, 1, 1).to(device)
    std = torch.tensor(transform_config['std']).view(1, 3, 1, 1).to(device)

    original_img = Image.open(img_path).convert('RGB')
    train_size = 378
    input_img = original_img.resize((train_size, train_size), Image.BICUBIC)

    img_tensor = torch.from_numpy(np.array(input_img)).permute(2, 0, 1).float() / 255.0
    img_tensor = (img_tensor.unsqueeze(0).to(device) - mean) / std

    # 2. Forward pass and obtain weights
    # For convenience, we directly extract the internal variables
    with torch.no_grad():
        # Here we simulate part of the logic in _forward_single
        B, _, H, W = img_tensor.shape
        x = model.model.prepare_tokens_with_masks(img_tensor)
        for blk in model.model.blocks:
            x = blk(x)
        x = model.model.norm(x)

        patch_tokens = x[:, 1:]  # [B, N, C]
        grid_h, grid_w = H // 14, W // 14

        # Prepare input for FeatureGrouping
        features = patch_tokens  # [B, HW, C]

        # Enter the logic of FeatureGrouping
        logits = model.featuregrouping.classifier(features)
        temperature = 1.5
        weights = F.softmax(logits / temperature, dim=-1)

        num_clusters = weights.shape[-1]
        attn_map = weights[0].transpose(0, 1).reshape(num_clusters, grid_h, grid_w)
        attn_map = F.interpolate(attn_map.unsqueeze(0), size=(train_size, train_size), mode='bilinear')[0]
        attn_map = attn_map.cpu().numpy()

    # 4. Plot heatmaps
    import os
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img_np = np.array(input_img)

    # Select the first 16 or all 64 clusters for visualization
    print(f"Generating heatmaps for {num_clusters} categories...")

    for i in range(num_clusters):
        mask = attn_map[i]

        # --- Spatial smoothing: use a larger sigma to reduce blocky artifacts ---
        mask = ndimage.gaussian_filter(mask, sigma=5.0)

        # --- Normalization strategy: use smooth stretching instead of percentile clipping ---
        # Subtract the minimum and divide by the maximum to normalize into [0, 1]
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

        # --- Apply smoothing again to reduce an overly sharp hot center ---
        mask = np.power(mask, 0.8)  # Gamma correction to enrich mid-tone colors

        # 5. Convert to color heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # 6. Reduce the heatmap weight during overlay
        # Blend ratio: 0.4 heatmap + 0.6 original image
        overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

        plt.imsave(f"{save_dir}/cluster_{i:02d}.png", overlay)

    print(f"Visualization completed. Results have been saved to: {save_dir}")


# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(model_name='dinov2_vitb14')

# Load the trained weights
state_dict = torch.load("weights_e95_0.9840.pth")
model.load_state_dict(state_dict)

# Run visualization
visualize_feature_grouping(model, img_path='/mnt/data1/liyong/University-Release/train/drone/0879/image-02.jpeg', device=device)
# visualize_feature_grouping(model, img_path='/mnt/data1/liyong/University-Release/train/satellite/0879/image-02.jpeg', device=device)