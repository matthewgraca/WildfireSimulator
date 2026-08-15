import torch
import torch.nn.functional as F

from tqdm import tqdm

import numpy as np

def _pad_to_multiple(tensor, multiple=32):
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return tensor, h, w
    # pad last dim (width) then second-last (height)
    padded = F.pad(tensor, (0, pad_w, 0, pad_h))
    return padded, h, w


class BurnerBatchProcessor:
    def __init__(
        self,
        burner,
        dt,
        eval,
        sampler=None,
        rng=None,
        device=None
    ):
        self.burner = burner
        self.dt = dt
        self.rng = rng
        self.eval = eval
        self.sampler = sampler
        self.device = device or torch.device('cpu')

    def __call__(self, pred, true, epoch, batch_idx, t):
        if not self.eval:
            self.rng.seed(epoch * 10_000 + batch_idx)

        N = true.size(0)
        H, W = true.shape[-2], true.shape[-1]

        # Move true to device once (stays on GPU for all operations)
        true = true.to(self.device)
        pred = pred.to(self.device)

        # --- Vectorized burn at time t (inputs) ---
        not_burnt_t = true[:, 1:2, :, :] > t  # (N, 1, H, W)
        burned_t = true.clone()
        burned_t[:, 0:1][not_burnt_t] = 0.0
        burned_t[:, 1:2][not_burnt_t] = 0.0

        # --- Vectorized burn at time t+dt (targets) ---
        not_burnt_dt = true[:, 1:2, :, :] > (t + self.dt)  # (N, 1, H, W)
        burned_dt = true.clone()
        burned_dt[:, 0:1][not_burnt_dt] = 0.0
        burned_dt[:, 1:2][not_burnt_dt] = 0.0

        # --- Scheduled sampling: pick pred or burned ground truth per sample ---
        if self.eval:
            # Crop pred back to original spatial dims so the time channel and
            # final padding behave identically to the non-eval path.
            in_frames = pred[:, :, :H, :W]
        else:
            prob = self.sampler.get_prob(epoch)
            use_pred_mask = torch.tensor(
                [self.rng.rand().item() < prob for _ in range(N)],
                dtype=torch.bool, device=self.device
            ).view(N, 1, 1, 1)
            # Crop pred to true's spatial dimensions for torch.where
            pred_cropped = pred[:, :, :H, :W]
            in_frames = torch.where(use_pred_mask, pred_cropped, burned_t)

        # --- Append time channel at original spatial dims (on device) ---
        t_channel = torch.full(
            (N, 1, H, W), t,
            device=self.device, dtype=true.dtype
        )
        inputs = torch.cat([in_frames, t_channel], dim=1)  # (N, 14, H, W)

        # --- Extract target (mask channel only of burned_dt) ---
        targets = burned_dt[:, 0:1]  # (N, 1, H, W)

        # --- Pad spatial dims to multiple of 32 ---
        inputs, _, _ = _pad_to_multiple(inputs, multiple=32)
        targets, _, _ = _pad_to_multiple(targets, multiple=32)

        return inputs, targets


class ForwardBurnTrainer:
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        train_loader,
        val_loader,
        train_batch_processor,
        val_batch_processor,
        callbacks=None,
        epochs=1,
        max_t=1,
        device=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_batch_processor = train_batch_processor
        self.val_batch_processor = val_batch_processor
        self.callbacks = callbacks or []
        self.epochs = epochs
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.current_epoch = 0
        self.max_t = max_t

        # Ensure batch processors use the same device as the trainer
        self.train_batch_processor.device = self.device
        self.val_batch_processor.device = self.device

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.current_epoch = checkpoint['epoch'] + 1
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def _train_epoch(self, epoch, total_epochs):
        self.model.train()
        total_loss = 0.0
        n_samples = len(self.train_loader.dataset)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        samples_seen = 0
        for batch_idx, batch in enumerate(pbar):
            num_steps = int(self.max_t / self.train_batch_processor.dt)

            # Initialize preds on device (avoids CPU→GPU transfer each step)
            N = batch.size(0)
            preds_padded = torch.zeros(N, 13, 512, 512, device=self.device)

            dt = self.train_batch_processor.dt
            for t in np.arange(dt, self.max_t, dt):
                inputs, targets = self.train_batch_processor(
                    preds_padded, batch, epoch=epoch, batch_idx=batch_idx, t=t
                )
                # inputs and targets are already on device (batch processor handles it)

                self.optimizer.zero_grad()

                pred_out = self.model(inputs)
                if isinstance(pred_out, (list, tuple)):
                    pred_out = pred_out[0]

                loss = self.loss_fn(pred_out, targets)
                loss.backward()
                self.optimizer.step()

                # Update preds for next time step (stay on device)
                preds_padded = inputs[:, :13, :, :].detach().clone()
                # Threshold predicted mask to binary
                pred_mask = (pred_out[:, 0:1].detach() > 0.5).float()
                preds_padded[:, 0:1, :, :] = pred_mask
                # Deterministic FAT update: newly burned pixels get arrival time = t + dt
                newly_burned = (pred_mask == 1) & (preds_padded[:, 1:2, :, :] == 0)
                preds_padded[:, 1:2, :, :][newly_burned] = t + dt

                total_loss += loss.item() * N / num_steps

            samples_seen += N
            pbar.set_postfix(loss=f"{total_loss / samples_seen:.4f}")

        return total_loss / n_samples

    def _validate(self, epoch, total_epochs):
        self.model.eval()
        total_loss = 0.0
        n_samples = len(self.val_loader.dataset)

        pbar = tqdm(self.val_loader, desc="Validating")
        samples_seen = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                preds_padded = None

                dt = self.val_batch_processor.dt
                num_steps = len(np.arange(dt, self.max_t, dt))
                batch_loss = 0.0

                for t in np.arange(dt, self.max_t, dt):
                    if preds_padded is None:
                        # Initialize with burned state at dt (gives model
                        # the initial fire seed, matching training behavior)
                        pred_input = batch.to(self.device).clone()
                        not_burnt = pred_input[:, 1:2, :, :] > dt
                        pred_input[:, 0:1][not_burnt] = 0.0
                        pred_input[:, 1:2][not_burnt] = 0.0
                    else:
                        pred_input = preds_padded

                    inputs, targets = self.val_batch_processor(
                        pred_input,
                        batch,
                        epoch=epoch,
                        batch_idx=batch_idx,
                        t=t,
                    )
                    # inputs and targets are already on device

                    pred_out = self.model(inputs)
                    if isinstance(pred_out, (list, tuple)):
                        pred_out = pred_out[0]

                    loss = self.loss_fn(pred_out, targets)
                    batch_loss += loss.item()

                    # Update preds for next time step (stay on device)
                    preds_padded = inputs[:, :13, :, :].detach().clone()
                    # Threshold predicted mask to binary
                    pred_mask = (pred_out[:, 0:1].detach() > 0.5).float()
                    preds_padded[:, 0:1, :, :] = pred_mask
                    # Deterministic FAT update: newly burned pixels get arrival time = t + dt
                    newly_burned = (pred_mask == 1) & (preds_padded[:, 1:2, :, :] == 0)
                    preds_padded[:, 1:2, :, :][newly_burned] = t + dt

                total_loss += (batch_loss / num_steps) * batch.size(0)
                samples_seen += batch.size(0)
                pbar.set_postfix(val_loss=f"{total_loss / samples_seen:.4f}")

        return total_loss / n_samples

    def fit(self):
        total_epochs = self.epochs
        for epoch in range(self.current_epoch, total_epochs):
            train_loss = self._train_epoch(epoch, total_epochs)

            # Capture per-component losses from training (last batch)
            train_components = {}
            for attr in ['last_bce', 'last_dice', 'last_focal', 'last_mask_loss', 'last_arrival_loss']:
                val = getattr(self.loss_fn, attr, None)
                if val is not None:
                    key = attr.replace('last_', '')
                    train_components[f'train_{key}'] = val

            val_loss = self._validate(epoch, total_epochs)
            metrics = {'train_loss': train_loss, 'val_loss': val_loss}
            metrics.update(train_components)

            # Capture per-component losses from validation (last batch)
            for attr in ['last_bce', 'last_dice', 'last_focal', 'last_mask_loss', 'last_arrival_loss']:
                val = getattr(self.loss_fn, attr, None)
                if val is not None:
                    key = attr.replace('last_', '')
                    metrics[f'val_{key}'] = val

            for cb in self.callbacks:
                cb.on_validation_end(epoch=epoch, metrics=metrics, model=self.model, optimizer=self.optimizer)
            self.current_epoch += 1

    def evaluate(self):
        val_loss = self._validate(epoch=0, total_epochs=1)
        return {'val_loss': val_loss}
