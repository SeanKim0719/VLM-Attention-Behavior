import os
import json
from pathlib import Path
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = str(BASE_DIR / "data" / "images")

def main():
    from clip_eval import CLIPEvaluator, BASE_PROMPTS
    from variants import save_variants

    evaluator = CLIPEvaluator()
    """
    for category in BASE_PROMPTS.keys():
        cat_path = Path(DATA_DIR) / category
        sample_img = next(cat_path.glob("*.jpg"), None)
        if sample_img:
            save_variants(str(sample_img), save_dir="results/variants_preview")
    """
    evaluator.evaluate_dataset(data_dir=DATA_DIR, grid_n=4)

    with open("results/raw_scores.json") as f:
        all_results = json.load(f)

    from analysis import (
        compute_category_stats,
        extract_grid_patterns,
        summarize_grid_patterns,
        plot_confidence_bars,
        plot_importance_map,
        plot_pattern_distribution,
        plot_summary_table,
        print_interpretation
    )

    all_stats           = {}
    all_grid_summaries  = {}
    all_per_image       = {}

    for category, results in all_results.items():
        all_stats[category] = compute_category_stats(results)

        per_image = extract_grid_patterns(results)
        all_per_image[category]      = per_image
        all_grid_summaries[category] = summarize_grid_patterns(per_image)

    Path("results").mkdir(exist_ok=True)

    plot_confidence_bars(all_stats)
    plot_pattern_distribution(all_grid_summaries)
    plot_summary_table(all_grid_summaries, all_stats)
    print_interpretation(all_grid_summaries, all_stats)

    for category, per_image in all_per_image.items():
        plot_importance_map(per_image, category)

    from gradcam_viz import run_attention_visualization
    run_attention_visualization()

if __name__ == "__main__":
    main()
