import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from pathlib import Path
import json

BASE_PROMPTS = {
    "cat":   "a photo of a cat",
    "dog":   "a photo of a dog",
    "fox":   "a photo of a fox",   
    "wolf":  "a photo of a wolf", 
    "tiger": "a photo of a tiger"  
}

VARIANT_PROMPTS = {
    "silhouette": {k: f"a silhouette of a {k}" for k in BASE_PROMPTS},
    "blurry":     {k: f"a blurry photo of a {k}" for k in BASE_PROMPTS}
}


class CLIPEvaluator:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

    def get_confidence(self, image_array: np.ndarray,
                       prompts: list) -> dict:
        image_pil = Image.fromarray(image_array.astype(np.uint8))
        inputs = self.processor(
            text=prompts, images=image_pil,
            return_tensors="pt", padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = outputs.logits_per_image.softmax(dim=1)[0].cpu().numpy()
        return {p: float(v) for p, v in zip(prompts, probs)}

    def evaluate_image(self, image_path: str,
                       true_category: str,
                       variants: dict) -> dict:
        all_prompts = list(BASE_PROMPTS.values())
        results = {}

        for variant_name, variant_img in variants.items():
            base_scores = self.get_confidence(variant_img, all_prompts)
            base_conf = base_scores[BASE_PROMPTS[true_category]]

            adapted_conf = None
            if variant_name in VARIANT_PROMPTS:
                adapted_prompts = list(
                    VARIANT_PROMPTS[variant_name].values()
                )
                adapted_scores = self.get_confidence(
                    variant_img, adapted_prompts
                )
                adapted_conf = adapted_scores[
                    VARIANT_PROMPTS[variant_name][true_category]
                ]

            results[variant_name] = {
                'base_confidence': base_conf,
                'adapted_confidence': adapted_conf,
                'all_categories': base_scores
            }
        return results

    def evaluate_grid_importance(self, image_path: str,
                                  true_category: str,
                                  n: int = 4) -> dict:
        from variants import create_grid_variants

        all_prompts = list(BASE_PROMPTS.values())
        target_prompt = BASE_PROMPTS[true_category]

        from variants import add_gray_padding
        import cv2
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        full_img = add_gray_padding(img)
        full_scores = self.get_confidence(full_img, all_prompts)
        full_conf = full_scores[target_prompt]

        grid_variants, grid_info = create_grid_variants(image_path, n)
        importance_map = {}

        for key, masked_img in grid_variants.items():
            scores = self.get_confidence(masked_img, all_prompts)
            masked_conf = scores[target_prompt]
            importance_map[key] = float(full_conf - masked_conf)

        pattern, decisive_regions = self._classify_pattern(
            importance_map, n
        )

        return {
            'full_confidence': full_conf,
            'importance_map': importance_map,
            'pattern': pattern,
            'decisive_regions': decisive_regions,
            'grid_size': n
        }

    def _classify_pattern(self, importance_map: dict,
                       n: int) -> tuple:
        values = list(importance_map.values())

        negative_values = [v for v in values if v < -0.02]
        positive_values = [v for v in values if v > 0]

        if negative_values:
            neg_abs_sum = sum(abs(v) for v in negative_values)
            pos_sum = sum(positive_values) if positive_values else 0
            total_magnitude = neg_abs_sum + pos_sum

            if total_magnitude > 0:
                interference_ratio = neg_abs_sum / total_magnitude
                if interference_ratio > 0.3:
                    decisive_regions = [
                        k for k, v in sorted(
                            importance_map.items(),
                            key=lambda x: x[1],
                            reverse=True
                        )[:3] if v > 0.05
                    ]
                    return "Interference-Present", decisive_regions

        if not positive_values:
            return "Undetermined", []

        total = sum(positive_values)
        if total < 0.01:
            return "Global-Distributed", []

        sorted_items = sorted(
            importance_map.items(),
            key=lambda x: x[1],
            reverse=True
        )

        top3_sum = sum(v for _, v in sorted_items[:3] if v > 0)
        concentration = top3_sum / total if total > 0 else 0

        decisive_regions = [k for k, v in sorted_items[:3] if v > 0.05]

        if concentration > 0.7:
            pattern = "Local-Focused"
        elif concentration < 0.4:
            pattern = "Global-Distributed"
        else:
            pattern = "Mixed"

        return pattern, decisive_regions

    def evaluate_dataset(self, data_dir: str = "data/images",
                          grid_n: int = 2) -> dict:
        from variants import create_variants
        from tqdm import tqdm

        data_path = Path(data_dir)
        all_results = {}

        for category in BASE_PROMPTS.keys():
            category_path = data_path / category
            if not category_path.exists():
                continue

            image_files = list(category_path.glob("*.jpg")) + \
                          list(category_path.glob("*.png"))

            all_results[category] = []

            for img_path in tqdm(image_files):
                variants = create_variants(str(img_path))
                variant_result = self.evaluate_image(
                    str(img_path), category, variants
                )

                grid_result = self.evaluate_grid_importance(
                    str(img_path), category, n=grid_n
                )

                combined = {
                    **variant_result,
                    'grid_analysis': grid_result,
                    'image_path': str(img_path)
                }
                all_results[category].append(combined)

        Path("results").mkdir(exist_ok=True)
        with open("results/raw_scores.json", "w") as f:
            json.dump(all_results, f, indent=2)

        return all_results