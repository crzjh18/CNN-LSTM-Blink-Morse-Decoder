import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import KMeans

# filepath: s:\VSCode Projects\MediaFace\EDA\eda_lstm_jsonl.py
DATA_PATH = "morse_sequences.jsonl"

def load_rows(path):
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except FileNotFoundError:
        print(f"Error: File '{path}' not found.")
    return rows

def main():
    rows = load_rows(DATA_PATH)
    if not rows:
        print("No data loaded.")
        return

    labels = [r.get("label", "?") for r in rows]
    durations = [r.get("raw_durations", []) for r in rows]
    lengths = [len(d) for d in durations]

    # Flatten all durations into a single array
    all_durs = np.concatenate([np.array(d) for d in durations if d])

    print("Total samples:", len(rows))
    print("Counts by label:", Counter(labels))
    
    if lengths:
        print("Sequence length (min/median/mean/max):",
              min(lengths), np.median(lengths), np.mean(lengths), max(lengths))
    
    # --- AUTOMATIC BLINK ANALYSIS ---
    dot_mean, dash_mean = 0, 0
    dot_min, dot_max = 0, 0
    dash_min, dash_max = 0, 0
    threshold = 0
    has_clusters = False
    
    if len(all_durs) > 1:
        # Reshape data for clustering
        X = all_durs.reshape(-1, 1)
        
        # Fit K-Means
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
        kmeans.fit(X)
        
        # Identify which cluster is which (Dot is smaller center)
        centers = kmeans.cluster_centers_.flatten()
        sorted_indices = np.argsort(centers)
        dot_idx = sorted_indices[0]
        dash_idx = sorted_indices[1]
        
        dot_mean = centers[dot_idx]
        dash_mean = centers[dash_idx]
        
        # Separate data
        dots = X[kmeans.labels_ == dot_idx]
        dashes = X[kmeans.labels_ == dash_idx]
        
        # Calculate Ranges
        if len(dots) > 0:
            dot_min, dot_max = dots.min(), dots.max()
        if len(dashes) > 0:
            dash_min, dash_max = dashes.min(), dashes.max()

        threshold = (dot_mean + dash_mean) / 2
        has_clusters = True
        
        print("\n" + "="*50)
        print(f" AUTOMATIC BLINK ANALYSIS")
        print("="*50)
        print(f" DOTS   | Avg: {dot_mean:.4f}s | Range: [{dot_min:.4f}s - {dot_max:.4f}s]")
        print(f" DASHES | Avg: {dash_mean:.4f}s | Range: [{dash_min:.4f}s - {dash_max:.4f}s]")
        print("-" * 50)
        print(f" SUGGESTED THRESHOLD: {threshold:.4f} s")
        print("="*50 + "\n")

    # --- Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Per-label counts
    lbls = list(Counter(labels).keys())
    if lbls:
        axes[0].bar(lbls, [Counter(labels)[l] for l in lbls], color="#2196f3")
        axes[0].set_title("Counts by label")
        axes[0].tick_params(axis="x", rotation=45)

    # 2. Sequence length distribution
    if lengths:
        axes[1].hist(lengths, bins=range(1, max(lengths)+2), color="#4caf50", align="left")
        axes[1].set_title("Blink count per sample")
        axes[1].set_xlabel("# blinks")

    # 3. Duration distribution
    if len(all_durs) > 0:
        axes[2].hist(all_durs, bins=40, color="#9c27b0", alpha=0.6, label="Data Distribution")
        axes[2].set_title("Blink Durations (s)")
        axes[2].set_xlabel("Duration (s)")
        
        if has_clusters:
            # Add vertical lines for Means
            axes[2].axvline(dot_mean, color='red', linestyle='--', linewidth=2, 
                            label=f'Dot Avg: {dot_mean:.2f}s')
            axes[2].axvline(dash_mean, color='blue', linestyle='--', linewidth=2, 
                            label=f'Dash Avg: {dash_mean:.2f}s')
            
            # Add shaded regions for Ranges (Min to Max)
            axes[2].axvspan(dot_min, dot_max, color='red', alpha=0.1, 
                            label=f'Dot Range: {dot_min:.2f}-{dot_max:.2f}s')
            axes[2].axvspan(dash_min, dash_max, color='blue', alpha=0.1, 
                            label=f'Dash Range: {dash_min:.2f}-{dash_max:.2f}s')

            # Threshold
            axes[2].axvline(threshold, color='green', linestyle=':', linewidth=2,
                            label=f'Threshold: {threshold:.2f}s')

        axes[2].legend(loc='upper right', fontsize='small')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()