import rasterio
import numpy as np
import yaml
import os
from glob import glob

def calculate_stats(tif_files, save_path):
    band_sums = np.zeros(6, dtype=np.float64)
    band_sumsq = np.zeros(6, dtype=np.float64)
    band_counts = np.zeros(6, dtype=np.int64)

    for tif_path in tif_files:
        with rasterio.open(tif_path) as src:
            for i in range(6):
                band = src.read(i + 1).astype(np.float64)
                band_sums[i] += band.sum()
                band_sumsq[i] += np.square(band).sum()
                band_counts[i] += band.size

    means = band_sums / band_counts
    stddevs = np.sqrt((band_sumsq / band_counts) - np.square(means))
    all_stats = {
        'mean': means.tolist(),
        'stddev': stddevs.tolist()
    }

    with open(save_path, 'w') as f:
        yaml.dump(all_stats, f, default_flow_style=False)

if __name__ == "__main__":
    base_path = os.getenv("BASE_PATH")
    data_path = os.path.join(base_path, "data")
    tif_dir = os.path.join(data_path, "train_samples", "images")
    save_path = os.path.join(data_path, "all_band_stats.yaml")

    tif_files = glob(os.path.join(tif_dir, '*.tif'))
    calculate_stats(tif_files, save_path)
