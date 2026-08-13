import argparse
import os

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from wildfire_simulator.models import MK_UNet_Regression
from wildfire_simulator.callbacks import ModelCheckpoint, TensorBoardCallback
from wildfire_simulator.forward_burn_process import ForwardBurnProcess
from wildfire_simulator.trainers import ForwardBurnTrainer, BurnerBatchProcessor
from wildfire_simulator.dataloader import TrialCollection, TrialFileLoader, WildfireDataLoader 
from wildfire_simulator.datasets import WildfireDataset, TransformedDataset
from wildfire_simulator.transforms import MinMaxPerChannel
from wildfire_simulator.scheduled_sampler import ScheduledSampler
from wildfire_simulator.losses import HybridLoss
from wildfire_simulator.utils import ScalarRNG
from wildfire_simulator.config import load_config

def main():
    parser = argparse.ArgumentParser(description="Train wildfire surrogate model")
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Directory to save checkpoints and training logs (default: current directory)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (default: config.yaml in project root)')
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    dataloader = WildfireDataLoader(TrialCollection(TrialFileLoader()))

    dataset = WildfireDataset(dataloader)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_set, test_set = random_split(
        dataset,
        [train_size, test_size],
        generator=generator
    )

    transform = MinMaxPerChannel(dataset.min_val, dataset.max_val)

    train_set = TransformedDataset(train_set, transform)
    test_set = TransformedDataset(test_set, transform)
    
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=16,
        shuffle=True,
        num_workers=4,
    )
    
    val_loader = DataLoader(
        dataset=test_set,
        batch_size=16,
        shuffle=False,
        drop_last=False,
        num_workers=4,
    )
    
    model = MK_UNet_Regression(
        in_channels=14,
        out_channels=2,
        channels=[16, 32, 64, 96, 160],
        final_activation='sigmoid'
    )
    
    checkpoint_cb = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        filepath=os.path.join(output_dir, 'checkpoints/best-model-{epoch:02d}-{val_loss:.2f}.pt')
    )

    train_writer = SummaryWriter(os.path.join(output_dir, "training/train"))
    val_writer = SummaryWriter(os.path.join(output_dir, "training/val"))

    tensorboard_cb = TensorBoardCallback(
        train_writer=train_writer,
        val_writer=val_writer
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        5e-4,
        weight_decay=1e-4
    )

    class ConstSampler():
        def __init__(self, prob):
            self.prob = prob
        def get_prob(self, epoch):
            return self.prob

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    burner = ForwardBurnProcess()
    sampler = ScheduledSampler(
        k=config['scheduled_sampling']['k'],
        t0=config['scheduled_sampling']['t0']
    )
    # sampler = ConstSampler(0.0)
    rng = ScalarRNG()
    train_batch_processor = BurnerBatchProcessor(
        burner=burner,
        dt=config['dt'],
        eval=False,
        sampler=sampler,
        rng=rng,
        device=device
    )
    val_batch_processor = BurnerBatchProcessor(
        burner=burner,
        dt=config['dt'],
        eval=True,
        device=device
    )
    
    trainer = ForwardBurnTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=HybridLoss(mask_weight=1.0, arrival_weight=10.0),
        train_loader=train_loader,
        val_loader=val_loader,
        train_batch_processor=train_batch_processor,
        val_batch_processor=val_batch_processor,
        callbacks=[checkpoint_cb, tensorboard_cb],
        epochs=100,
        max_t=config['max_t'],
        device=device,
    )

    trainer.fit()

    # trainer.load_checkpoint("./checkpoints/best-model-04-0.00.pt")
    # print(trainer.evaluate())

if __name__ == "__main__":
    main()
