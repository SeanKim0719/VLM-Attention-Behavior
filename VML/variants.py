import cv2
import numpy as np
from PIL import Image
from pathlib import Path


def add_gray_padding(image: np.ndarray, 
                     target_size: tuple = (224, 224)) -> np.ndarray:
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h))

    canvas = np.full((target_h, target_w, 3), 128, dtype=np.uint8)
    
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    canvas[y_offset:y_offset+new_h, 
           x_offset:x_offset+new_w] = resized
    
    return canvas


def create_variants(image_path: str) -> dict:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Can't find image: {image_path}. Check the path.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    variants = {}

    variants['full'] = add_gray_padding(img)
    
    top = img[:h//2, :, :]
    variants['top_half'] = add_gray_padding(top)
    
    bottom = img[h//2:, :, :]
    variants['bottom_half'] = add_gray_padding(bottom)
    
    center = img[h//4:3*h//4, w//4:3*w//4, :]
    variants['center'] = add_gray_padding(center)
    
    edges_only = img.copy()
    edges_only[h//4:3*h//4, w//4:3*w//4, :] = 128
    variants['edges_only'] = add_gray_padding(edges_only)
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    silhouette = np.full_like(img, 255)
    silhouette[edges > 0] = [0, 0, 0]
    variants['silhouette'] = add_gray_padding(silhouette)
    
    blurry = cv2.GaussianBlur(img, (51, 51), 0)
    variants['blurry'] = add_gray_padding(blurry)
    
    return variants

def create_grid_variants(image_path: str, n: int = 4) -> dict:
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    cell_h = h // n
    cell_w = w // n

    grid_variants = {}

    for row in range(n):
        for col in range(n):
            masked = img.copy()

            y1, y2 = row * cell_h, (row + 1) * cell_h
            x1, x2 = col * cell_w, (col + 1) * cell_w
            masked[y1:y2, x1:x2] = 128

            key = f"grid_{row}_{col}"
            grid_variants[key] = add_gray_padding(masked)

    return grid_variants, (n, cell_h, cell_w)

def save_variants(image_path: str, 
                  save_dir: str = "data/variants"):
    path = Path(image_path)
    variants = create_variants(image_path)
    
    save_path = Path(save_dir) / path.stem
    save_path.mkdir(parents=True, exist_ok=True)
    
    for name, img in variants.items():
        out = Image.fromarray(img.astype(np.uint8))
        out.save(save_path / f"{name}.jpg")
    
    print(f"Variants saved: {save_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        save_variants(sys.argv[1])
    else:
        print("Usage: python variants.py <imagepath>")