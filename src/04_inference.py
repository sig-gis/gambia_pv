import numpy as np
import os
import rasterio
import torch

from torch.utils.data import DataLoader

from utils.unet import UNet
from utils.dataset import InferenceDataset
from utils.utils import find_lowest_loss


def inference(model, tif_path, stats_file, output_path, batch_size):
    dataset = InferenceDataset(tif_path, stats_file)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with rasterio.open(tif_path) as src:
        meta = src.meta.copy()
        height, width = src.height, src.width
        transform = src.transform
        crs = src.crs

    prediction_sum = np.zeros((height, width), dtype=np.float32)
    prediction_count = np.zeros((height, width), dtype=np.uint16)
    with torch.no_grad():
        for step, (images, idxes) in enumerate(loader):
            print(f"Processing batch {step}/{len(loader)}")
            images = images.to(torch.device("cuda"))
            preds = model(images)
            preds = torch.sigmoid(preds).squeeze(1).cpu().numpy()  # Shape: (B, H, W)

            for i, idx in enumerate(idxes):
                win = dataset.windows[idx]
                row_off, col_off = win.row_off, win.col_off
                h, w = win.height, win.width
                prediction_sum[row_off:row_off+h, col_off:col_off+w] += preds[i]
                prediction_count[row_off:row_off+h, col_off:col_off+w] += 1
    prediction_count[prediction_count == 0] = 1
    averaged_prediction = prediction_sum / prediction_count
    meta.update({
        "count": 1,
        "dtype": "float32"
    })

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(averaged_prediction[np.newaxis, :, :])


if __name__ == "__main__":
    base_path = os.getenv("BASE_PATH")
    data_path = os.path.join(base_path, "data")
    ckpt_path = os.path.join(base_path, "checkpoints")
    
    experiment_name = os.getenv("EXPERIMENT_NAME")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=6, out_channels=1).to(device)
    best_ckpt_path = find_lowest_loss(os.path.join(ckpt_path, f"{experiment_name}/*.pt"))
    ckpt = torch.load(best_ckpt_path, map_location=device)

    model.load_state_dict(ckpt)
    model.eval()
    
    tif_path = os.path.join(data_path, "pleides_RGBNED.tif")
    stats_file = os.path.join(data_path, "all_band_stats.yaml")
    output_path = os.path.join(data_path, f"{experiment_name}.tif")
    
    inference(model, tif_path, stats_file, output_path, batch_size=512)
    