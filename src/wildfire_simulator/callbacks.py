import torch
import os

class ModelCheckpoint:
    def __init__(self, monitor, mode, filepath):
        self.monitor = monitor
        self.mode = mode
        self.filepath_template = filepath
        self.best_metric = None

        if self.mode == 'min':
            self.compare = lambda current, best: current < best
        elif self.mode == 'max':
            self.compare = lambda current, best: current > best
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def on_validation_end(self, epoch, metrics, model, optimizer):
        current = metrics.get(self.monitor)
        if current is None:
            return
        if self.best_metric is None or self.compare(current, self.best_metric):
            self.best_metric = current
            path = self.filepath_template.format(epoch=epoch, val_loss=current)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, path)

class TensorBoardCallback:
    def __init__(self, train_writer, val_writer):
        self.train_writer = train_writer
        self.val_writer = val_writer

    def on_validation_end(self, epoch, metrics, model, optimizer):
        self.train_writer.add_scalar("Loss", metrics['train_loss'], epoch)
        self.val_writer.add_scalar("Loss", metrics['val_loss'], epoch)

        # Log per-channel losses if available (from HybridLoss)
        if 'train_mask_loss' in metrics:
            self.train_writer.add_scalar("Loss/mask_bce", metrics['train_mask_loss'], epoch)
        if 'train_arrival_loss' in metrics:
            self.train_writer.add_scalar("Loss/arrival_mse", metrics['train_arrival_loss'], epoch)
        if 'val_mask_loss' in metrics:
            self.val_writer.add_scalar("Loss/mask_bce", metrics['val_mask_loss'], epoch)
        if 'val_arrival_loss' in metrics:
            self.val_writer.add_scalar("Loss/arrival_mse", metrics['val_arrival_loss'], epoch)
