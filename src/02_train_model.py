import comet_ml
import torch
import torch.optim as optim
import os
import heapq


from comet_ml import Experiment
from torch.utils.data import DataLoader
from torchmetrics.functional.classification import (binary_accuracy,
                                                    binary_precision,
                                                    binary_jaccard_index,
                                                    binary_recall)
from torchmetrics.functional.image import structural_similarity_index_measure
from typing import Literal

from utils.loss import DiceLoss, FocalLoss, DiceFocalLoss
from utils.unet import UNet
from utils.dataset import PVDataset



def train(data_path, stats_file, project_name, experiment_name, ckpt_dir,
          epochs=1024, batch_size=16, learning_rate=1e-5, criterion: Literal["dice", "focal", "dice_focal", "bce"] = "dice"):
    comet_api_key = os.environ["COMET_API_KEY"]
    workspace = os.environ["COMET_WORKSPACE"]

    experiment = Experiment(
        api_key=comet_api_key,
        project_name=project_name,
        workspace=workspace,
        auto_metric_logging=False,
    )
    experiment.set_name(experiment_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=6, out_channels=1).to(device)

    train_dataset = PVDataset(data_path, stats_file, split="train_samples", image_size=224)
    test_dataset = PVDataset(data_path, stats_file, split="test_samples", image_size=224)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    match criterion:
        case "dice":
            criterion = DiceLoss()
        case "focal":
            criterion = FocalLoss()
        case "dice_focal":
            criterion = DiceFocalLoss()
        case "bce":
            criterion = torch.nn.BCEWithLogitsLoss()
        case _:
            raise ValueError(f"Unknown criterion: {criterion}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    experiment.log_parameters({
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "input_channels": 6,
        "image_size": 224,
        "criterion": criterion,
        "optimizer": "Adam",
    })

    best_ckpts = []
    experiment_ckpt_dir = os.path.join(ckpt_dir, experiment_name)
    os.makedirs(experiment_ckpt_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        for step, (images, masks, _) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)

            preds = model(images)
            loss = criterion(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            experiment.log_metric("train_loss", loss.item(), step=step*(epoch+1), epoch=epoch)

        model.eval()
        total_test_loss = 0
        test_ssim = 0
        test_jaccard = 0
        test_accuracy = 0
        test_precision = 0
        test_recall = 0

        with torch.no_grad():
            for images, masks, _ in test_loader:
                images, masks = images.to(device), masks.to(device)

                preds = model(images)
                loss = criterion(preds, masks)
                total_test_loss += loss.item()

                preds_sigmoid = torch.sigmoid(preds)
                test_ssim += structural_similarity_index_measure(preds_sigmoid, masks).item()
                test_jaccard += binary_jaccard_index(preds_sigmoid, masks).item()
                test_accuracy += binary_accuracy(preds_sigmoid, masks.int()).item()
                test_precision += binary_precision(preds_sigmoid, masks.int()).item()
                test_recall += binary_recall(preds_sigmoid, masks.int()).item()

        avg_test_loss = total_test_loss / len(test_loader)
        avg_test_ssim = test_ssim / len(test_loader)
        avg_test_jaccard = test_jaccard / len(test_loader)
        avg_test_accuracy = test_accuracy / len(test_loader)
        avg_test_precision = test_precision / len(test_loader)
        avg_test_recall = test_recall / len(test_loader)

        experiment.log_metric("test_loss", avg_test_loss, step=step*(epoch+1), epoch=epoch)
        experiment.log_metric("test_ssim", avg_test_ssim, step=step*(epoch+1), epoch=epoch)
        experiment.log_metric("test_jaccard", avg_test_jaccard, step=step*(epoch+1), epoch=epoch)
        experiment.log_metric("test_accuracy", avg_test_accuracy, step=step*(epoch+1), epoch=epoch)
        experiment.log_metric("test_precision", avg_test_precision, step=step*(epoch+1), epoch=epoch)
        experiment.log_metric("test_recall", avg_test_recall, step=step*(epoch+1), epoch=epoch)

        # Save top 4 checkpoints
        ckpt_path = os.path.join(experiment_ckpt_dir, f"epoch{epoch:04d}_loss{avg_test_loss:.4f}.pt")
        if len(best_ckpts) < 4:
            torch.save(model.state_dict(), ckpt_path)
            heapq.heappush(best_ckpts, (-avg_test_loss, ckpt_path))
        else:
            worst_loss, worst_path = best_ckpts[0]
            if avg_test_loss < -worst_loss:
                torch.save(model.state_dict(), ckpt_path)
                os.remove(worst_path)
                heapq.heapreplace(best_ckpts, (-avg_test_loss, ckpt_path))

    experiment.end()


if __name__ == "__main__":
    base_path = os.getenv("BASE_PATH")
    data_path = os.path.join(base_path, "data")
    stats_file = os.path.join(data_path, "all_band_stats.yaml")

    project_name = "gambia_pv"
    experiment_name = os.getenv("EXPERIMENT_NAME")
    ckpt_dir = os.path.join(base_path, "checkpoints")

    train(data_path, stats_file, project_name, experiment_name, ckpt_dir)
