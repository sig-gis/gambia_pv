import os
import rasterio
import torch

from torch.utils.data import DataLoader
from torchmetrics.functional.classification import (binary_accuracy,
                                                    binary_precision,
                                                    binary_jaccard_index,
                                                    binary_recall)
from torchmetrics.functional.image import structural_similarity_index_measure
from rasterio.windows import Window

from utils.loss import DiceLoss
from utils.unet import UNet
from utils.dataset import PVDataset
from utils.utils import find_lowest_loss


def inference(model, data_path, stats_file, output_path, batch_size=32):
    test_dataset = PVDataset(data_path, stats_file, split="test_samples", image_size=224)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    criterion = DiceLoss()
    with torch.no_grad():
        total_test_loss = 0
        test_ssim = 0
        test_jaccard = 0
        test_accuracy = 0
        test_precision = 0
        test_recall = 0
        for images, masks, idxes in test_loader: 
            images = images.to(device)
            masks = masks.to(device)
            preds = model(images)
            preds = torch.sigmoid(preds)
            
            loss = criterion(preds, masks)
            total_test_loss += loss.item()
            test_ssim += structural_similarity_index_measure(preds, masks).item()
            test_jaccard += binary_jaccard_index(preds, masks).item()
            test_accuracy += binary_accuracy(preds, masks.int()).item()
            test_precision += binary_precision(preds, masks.int()).item()
            test_recall += binary_recall(preds, masks.int()).item()

            for i in range(len(preds)):
                pred = preds[i]
                idx = idxes[i]
                path = test_dataset.image_paths[idx]
                base_name = os.path.basename(path)
                with rasterio.open(path) as src:
                    full_height, full_width = src.height, src.width
                    crop_h = crop_w = 224
                    top = (full_height - crop_h) // 2
                    left = (full_width - crop_w) // 2
                    window = Window(left, top, crop_w, crop_h)
                    
                    out_meta = src.meta.copy()
                    out_meta.update({
                        "height": 224,
                        "width": 224,
                        "count": 1,
                        "transform": src.window_transform(window)
                    })
                out_path = os.path.join(output_path, base_name)
                with rasterio.open(out_path, "w", **out_meta) as dest:
                    dest.write(pred.cpu().numpy())
                avg_test_loss = total_test_loss / len(test_loader)
        avg_test_ssim = test_ssim / len(test_loader)
        avg_test_jaccard = test_jaccard / len(test_loader)
        avg_test_accuracy = test_accuracy / len(test_loader)
        avg_test_precision = test_precision / len(test_loader)
        avg_test_recall = test_recall / len(test_loader)
        
        print(f"Test Loss: {avg_test_loss:.4f}")
        print(f"Test SSIM: {avg_test_ssim:.4f}")
        print(f"Test Jaccard: {avg_test_jaccard:.4f}")
        print(f"Test Accuracy: {avg_test_accuracy:.4f}")
        print(f"Test Precision: {avg_test_precision:.4f}")
        print(f"Test Recall: {avg_test_recall:.4f}")
        

if __name__ == "__main__":
    base_path = os.getenv("BASE_PATH")
    data_path = os.path.join(base_path, "data")
    ckpt_path = os.path.join(base_path, "checkpoints")
    
    experiment_name = os.getenv("EXPERIMENT_NAME")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=6, out_channels=1).to(device)
    best_ckpt_path = find_lowest_loss(os.path.join(ckpt_path, f"{experiment_name}/*.pt"))
    print(f"Loading best checkpoint from {best_ckpt_path}")
    
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    
    stats_file = os.path.join(data_path, "all_band_stats.yaml")
    output_path = os.path.join(data_path, "test_model", experiment_name)

    os.makedirs(output_path, exist_ok=True)
    inference(model, data_path, stats_file, output_path, batch_size=32)
