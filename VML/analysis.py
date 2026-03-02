import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

VARIANT_ORDER = [
    'full', 'center', 'top_half',
    'bottom_half', 'edges_only', 'blurry', 'silhouette'
]
VARIANT_LABELS = {
    'full': 'Full', 'center': 'Center 50%',
    'top_half': 'Top Half', 'bottom_half': 'Bottom Half',
    'edges_only': 'Edges Only', 'blurry': 'Blurry',
    'silhouette': 'Silhouette'
}
PATTERN_COLORS = {
    'Local-Focused':       '#FFCDD2', 
    'Global-Distributed':  '#C8E6C9', 
    'Mixed':               '#FFF9C4',
    'Interference-Present' : '#CE93D8',
    'Undetermined' :        '#E0E0E0'
}

def compute_category_stats(category_results: list) -> dict:
    stats = {v: [] for v in VARIANT_ORDER}
    for image_result in category_results:
        for variant in VARIANT_ORDER:
            if variant in image_result:
                conf = image_result[variant]['base_confidence']
                stats[variant].append(conf)
    return {
        v: {'mean': np.mean(s), 'std': np.std(s)}
        for v, s in stats.items() if s
    }

def classify_variant_pattern(stats: dict) -> dict:
    full_mean = stats['full']['mean']
    sil_mean  = stats['silhouette']['mean']
    ctr_mean  = stats['center']['mean']
    blr_mean  = stats.get('blurry', {}).get('mean', 0)
    drop_rate = (full_mean - sil_mean) / full_mean

    if drop_rate > 0.5:
        pattern = "Bottom-Up"
    elif drop_rate < 0.3 and ctr_mean / full_mean > 0.8:
        pattern = "Top-Down"
    else:
        pattern = "Hybrid"

    return {
        'pattern': pattern,
        'full_mean': full_mean, 'silhouette_mean': sil_mean,
        'center_mean': ctr_mean, 'blurry_mean': blr_mean,
        'drop_rate': drop_rate
    }

def extract_grid_patterns(category_results: list) -> list:
    per_image = []
    for result in category_results:
        if 'grid_analysis' not in result:
            continue
        g = result['grid_analysis']
        per_image.append({
            'image_path':      result.get('image_path', ''),
            'full_confidence': g['full_confidence'],
            'pattern':         g['pattern'],
            'decisive_regions': g['decisive_regions'],
            'importance_map':  g['importance_map'],
            'grid_size':       g['grid_size']
        })
    return per_image


def summarize_grid_patterns(per_image: list) -> dict:
    from collections import Counter
    counts = Counter(p['pattern'] for p in per_image)
    total  = len(per_image)
    return {
        pattern: {'count': count, 'ratio': count / total}
        for pattern, count in counts.items()
    }

def plot_confidence_bars(all_stats: dict,
                          save_path: str = "results/confidence_bars.png"):
    categories = list(all_stats.keys())
    n_cats = len(categories)
    fig, axes = plt.subplots(1, n_cats, figsize=(5 * n_cats, 6), sharey=True)
    if n_cats == 1:
        axes = [axes]

    colors = {
        'full': '#2196F3', 'center': '#4CAF50',
        'top_half': '#8BC34A', 'bottom_half': '#CDDC39',
        'edges_only': '#FF9800', 'blurry': '#FF5722',
        'silhouette': '#9C27B0'
    }

    for ax, category in zip(axes, categories):
        stats  = all_stats[category]
        pinfo  = classify_variant_pattern(stats)
        valid  = [v for v in VARIANT_ORDER if v in stats]
        means  = [stats[v]['mean'] for v in valid]
        stds   = [stats[v]['std']  for v in valid]
        labels = [VARIANT_LABELS[v] for v in valid]
        bcolors = [colors[v] for v in valid]

        bars = ax.bar(range(len(valid)), means, yerr=stds,
                      color=bcolors, alpha=0.85, capsize=4)
        ax.set_xticks(range(len(valid)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(
            f"{category.upper()}\n"
            f"Pattern: {pinfo['pattern']} (drop:{pinfo['drop_rate']:.0%})",
            fontsize=11, fontweight='bold'
        )
        ax.set_ylabel("CLIP Confidence Score" if ax == axes[0] else "")

        full_conf = stats['full']['mean']
        for bar, m in zip(bars, means):
            if m < full_conf * 0.5:
                bar.set_edgecolor('red')
                bar.set_linewidth(2)

    plt.suptitle("CLIP Confidence Across Input Variants\n"
                 "(Red border = >50% drop from Full Image)",
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"save: {save_path}")


def plot_importance_map(per_image_results: list,
                         category: str,
                         save_dir: str = "results/importance_maps"):
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for idx, result in enumerate(per_image_results):
        n = result['grid_size']
        imp_map = result['importance_map']
        pattern = result['pattern']
        img_name = Path(result['image_path']).stem

        grid = np.zeros((n, n))
        for row in range(n):
            for col in range(n):
                key = f"grid_{row}_{col}"
                grid[row, col] = imp_map.get(key, 0)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        try:
            from PIL import Image as PILImage
            img = PILImage.open(result['image_path']).convert("RGB")
            axes[0].imshow(img)
        except Exception:
            axes[0].text(0.5, 0.5, "Image not found",
                         ha='center', va='center')
        axes[0].set_title(f"Original: {img_name}", fontsize=12)
        axes[0].axis('off')

        im = axes[1].imshow(grid, cmap='Reds',
                             vmin=min(0, grid.min()),
                             vmax=max(0.1, grid.max()))
        plt.colorbar(im, ax=axes[1], label='Confidence Drop')

        for region in result['decisive_regions']:
            _, r, c = region.split('_')
            rect = patches.Rectangle(
                (int(c) - 0.5, int(r) - 0.5), 1, 1,
                linewidth=3, edgecolor='blue', facecolor='none'
            )
            axes[1].add_patch(rect)

        axes[1].set_xticks(range(n))
        axes[1].set_yticks(range(n))
        axes[1].set_title(
            f"Importance Map\nPattern: {pattern}\n"
            f"(Blue border = decisive region)",
            fontsize=12
        )

        plt.suptitle(
            f"Grid Importance Analysis: {category.upper()} #{idx+1}",
            fontsize=14, fontweight='bold'
        )
        save_path = f"{save_dir}/{category}_{img_name}_importance.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"save: {save_path}")


def plot_pattern_distribution(all_grid_summaries: dict,
                               save_path: str = "results/pattern_distribution.png"):
    categories  = list(all_grid_summaries.keys())
    pattern_types = ['Local-Focused', 'Global-Distributed',
                     'Mixed', 'Interference-Present', 'Undetermined']
    bar_colors  = ['#EF9A9A', '#A5D6A7', '#FFF59D', '#CE93D8', '#E0E0E0']

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(categories))
    width = 0.15

    for i, (ptype, color) in enumerate(zip(pattern_types, bar_colors)):
        ratios = [
            all_grid_summaries[cat].get(ptype, {}).get('ratio', 0)
            for cat in categories
        ]
        ax.bar(x + i * width, ratios, width,
               label=ptype, color=color, edgecolor='gray', linewidth=0.5)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([c.upper() for c in categories])
    ax.set_ylabel("Ratio of Images")
    ax.set_ylim(0, 1.0)
    ax.legend(loc='upper right')
    ax.set_title(
        "Per-Image Pattern Distribution by Category\n"
        "(Local-Focused = specific region matters, "
        "Global-Distributed = whole structure matters)",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"save: {save_path}")


def plot_summary_table(all_grid_summaries: dict,
                        all_stats: dict,
                        save_path: str = "results/pattern_summary.png"):
    categories = list(all_stats.keys())
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.axis('off')

    col_labels = [
        'Category',
        'Variant Pattern',
        'Drop Rate',
        'Local-Focused',
        'Global-Distributed',
        'Mixed',
        'Interference',
        'Dominant Pattern'
    ]

    table_data = []
    row_colors = []

    for cat in categories:
        vp = classify_variant_pattern(all_stats[cat])
        gs = all_grid_summaries.get(cat, {})

        local_r  = gs.get('Local-Focused',      {}).get('ratio', 0)
        global_r = gs.get('Global-Distributed', {}).get('ratio', 0)
        mixed_r  = gs.get('Mixed',              {}).get('ratio', 0)
        interf_r    = gs.get('Interference-Present', {}).get('ratio', 0)

        dominant = max(
            [('Local-Focused', local_r),
             ('Global-Distributed', global_r),
             ('Mixed', mixed_r),
             ('Interference-Present', interf_r)],
            key=lambda x: x[1]
        )[0]

        table_data.append([
            cat.upper(),
            vp['pattern'],
            f"{vp['drop_rate']:.0%}",
            f"{local_r:.0%}",
            f"{global_r:.0%}",
            f"{mixed_r:.0%}",
            f"{interf_r:.0%}",
            dominant
        ])
        row_colors.append(
            PATTERN_COLORS.get(dominant, 'white')
        )

    table = ax.table(
        cellText=table_data, colLabels=col_labels,
        loc='center', cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)

    for i, color in enumerate(row_colors):
        for j in range(len(col_labels)):
            table[i + 1, j].set_facecolor(color)

    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#1F3864')
        table[0, j].set_text_props(color='white', fontweight='bold')

    plt.title("VLM Reasoning Pattern Summary\n"
              "(Variant-level + Per-Image Grid Analysis)",
              fontsize=13, fontweight='bold', pad=20)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"save: {save_path}")


def print_interpretation(all_grid_summaries: dict,
                          all_stats: dict):
    print("\n" + "="*60)
    print("RESULTS INTERPRETATION")
    print("="*60)

    for cat in all_stats.keys():
        vp = classify_variant_pattern(all_stats[cat])
        gs = all_grid_summaries.get(cat, {})

        print(f"\n[{cat.upper()}]")
        print(f"  Variant Pattern : {vp['pattern']} "
              f"(silhouette drop {vp['drop_rate']:.0%})")

        for ptype, info in gs.items():
            print(f"  {ptype:<25}: "
                  f"{info['count']}장 ({info['ratio']:.0%})")

        local_r  = gs.get('Local-Focused',      {}).get('ratio', 0)
        global_r = gs.get('Global-Distributed', {}).get('ratio', 0)