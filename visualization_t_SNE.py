import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import numpy as np
from model import Model


def visualize_feature_distribution(model, img_path, device='cuda', title="Feature Distribution"):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.unicode_minus"] = False
    model.eval()
    model.to(device)

    transform_config = model.get_config()
    mean = torch.tensor(transform_config['mean']).view(1, 3, 1, 1).to(device)
    std = torch.tensor(transform_config['std']).view(1, 3, 1, 1).to(device)

    from PIL import Image
    original_img = Image.open(img_path).convert('RGB')
    train_size = 378
    input_img = original_img.resize((train_size, train_size), Image.BICUBIC)

    img_tensor = torch.from_numpy(np.array(input_img)).permute(2, 0, 1).float() / 255.0
    img_tensor = (img_tensor.unsqueeze(0).to(device) - mean) / std

    with torch.no_grad():
        B, _, H, W = img_tensor.shape
        model.train()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            _, local_features = model._forward_single(img_tensor)
        model.eval()

    # (64, 128)
    features_np = local_features[0].cpu().numpy()

    tsne = TSNE(n_components=2, init='pca', learning_rate='auto', perplexity=15, random_state=42)
    features_2d = tsne.fit_transform(features_np)

    plt.figure(figsize=(6, 6))

    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1],
                          c=range(len(features_2d)), cmap='nipy_spectral',
                          s=100, edgecolors='white', alpha=0.8)

    # plt.colorbar(scatter, label='Cluster ID')
    # plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.8)
    # plt.tick_params(axis='both', labelsize=12)
    # plt.xlabel("t-SNE Dimension 1")
    # plt.ylabel("t-SNE Dimension 2")

    save_path = f"{title.replace(' ', '_')}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
    print(f"Feature distribution plot has been saved to: {save_path}")
    plt.show()

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(model_name='dinov2_vitb14')

# Load the trained weights
state_dict = torch.load("weights_e95_0.9840.pth", map_location=device)
# state_dict = torch.load("weights_end.pth", map_location=device)
model.load_state_dict(state_dict)

# Visualize the feature distribution
visualize_feature_distribution(model,'/mnt/data1/liyong/University-Release/train/drone/0879/image-02.jpeg', title='With_Self-supervised_Constraint')
