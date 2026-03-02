import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from pathlib import Path


def get_clip_attention_map(model, processor, 
                            image_path: str,
                            text_prompt: str) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    
    inputs = processor(
        text=[text_prompt],
        images=image,
        return_tensors="pt",
        padding=True
    )
    
    attention_weights = []
    
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            target_output = output[0]
        else:
            target_output = output
            
        attention_weights.append(
            target_output.detach().cpu()
        )
    
    hooks = []
    for layer in model.vision_model.encoder.layers:
        hook = layer.self_attn.register_forward_hook(hook_fn)
        hooks.append(hook)
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    for hook in hooks:
        hook.remove()
    
    if not attention_weights:
        return None
    
    rollout = torch.eye(attention_weights[0].shape[-1])
    
    for attn in attention_weights:
        attn_avg = attn[0].mean(dim=0)
        attn_avg = attn_avg + torch.eye(attn_avg.shape[-1])
        attn_avg = attn_avg / attn_avg.sum(dim=-1, keepdim=True)
        rollout = torch.matmul(attn_avg, rollout)
    
    cls_attention = rollout[0, 1:] 
    
    num_patches = len(cls_attention)
    grid_size = int(np.sqrt(num_patches))
    
    if grid_size * grid_size != num_patches:
        grid_size = int(np.ceil(np.sqrt(num_patches)))
        padded_attention = np.zeros(grid_size * grid_size)
        padded_attention[:num_patches] = cls_attention.numpy()
        attention_map = padded_attention.reshape(grid_size, grid_size)
    else:
        attention_map = cls_attention.reshape(grid_size, grid_size).numpy()
    
    img_array = np.array(image)
    attention_resized = np.array(
        Image.fromarray(
            (attention_map * 255).astype(np.uint8)
        ).resize(
            (img_array.shape[1], img_array.shape[0]),
            Image.BILINEAR
        )
    ) / 255.0
    
    return attention_resized


def visualize_attention(image_path: str,
                         true_category: str,
                         model, processor,
                         save_dir: str = "results/gradcam"):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    image = Image.open(image_path).convert("RGB")
    img_array = np.array(image)
    
    prompt = f"a photo of a {true_category}"
    attention_map = get_clip_attention_map(
        model, processor, image_path, prompt
    )
    
    if attention_map is None:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_array)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis('off')
    
    im = axes[1].imshow(attention_map, cmap='jet')
    axes[1].set_title(
        f"Attention Map\n(prompt: '{prompt}')", 
        fontsize=12
    )
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    
    axes[2].imshow(img_array)
    axes[2].imshow(
        attention_map, 
        cmap='jet', 
        alpha=0.5
    )
    axes[2].set_title("Overlay", fontsize=12)
    axes[2].axis('off')
    
    img_name = Path(image_path).stem
    plt.suptitle(
        f"CLIP Attention Visualization: {true_category.upper()}",
        fontsize=14, fontweight='bold'
    )
    
    save_path = f"{save_dir}/{img_name}_attention.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"save: {save_path}")
    return attention_map


def run_attention_visualization(
        data_dir: str = "data/images",
        categories: list = None):
    from transformers import CLIPModel, CLIPProcessor
    
    model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32"
    )
    processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )
    model.eval()
    
    if categories is None:
        categories = ["cat", "dog", "fox", "wolf", "tiger"]
    
    for category in categories:
        cat_path = Path(data_dir) / category
        if not cat_path.exists():
            continue
        
        images = list(cat_path.glob("*.jpg"))
        
        for img_path in images:
            visualize_attention(
                str(img_path), 
                category, 
                model, 
                processor
            )

if __name__ == "__main__":
    run_attention_visualization()