import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import torch.nn as nn
from pathlib import Path
import os
import shutil
import re
import numpy as np

from wildfire_simulator.callbacks import ModelCheckpoint, TensorBoardCallback
from wildfire_simulator.datasets import WildfireDataset, TransformedDataset
from wildfire_simulator.transforms import MinMaxPerChannel
from wildfire_simulator.forward_burn_process import ForwardBurnProcess
from wildfire_simulator.models import MK_UNet_Regression
from wildfire_simulator.trainers import ForwardBurnTrainer, BurnerBatchProcessor
from wildfire_simulator.scheduled_sampler import ScheduledSampler
from wildfire_simulator.utils import ScalarRNG

def test_batch_processor(dataloader):
    dataset = WildfireDataset(dataloader)

    burner = ForwardBurnProcess()

    sampler = ScheduledSampler(k=0.1, t0=40)

    class ConstSampler():
        def __init__(self, prob):
            self.prob = prob
        def get_prob(self, epoch):
            return self.prob

    class DummyScalarRNG:
        def __init__(self, constant_value = 0.5):
            self.constant_value = constant_value
            self.last_seed_used = None

        def seed(self, seed_value):
            self.last_seed_used = seed_value

        def rand(self):
            return torch.tensor(self.constant_value, dtype=torch.float32)

    rng = DummyScalarRNG(0.5)
    batch_processor = BurnerBatchProcessor(
        burner=burner,
        dt=30,
        eval=False,
        sampler=ConstSampler(0.4),
        rng=rng
    )
    assert batch_processor.dt == 30

    pred = torch.stack([dataset[1], dataset[0]])
    true = torch.stack([dataset[0], dataset[1]])

    input_tensor, output_tensor = batch_processor(pred, true, epoch=6, batch_idx=7, t=60)
    assert rng.last_seed_used == 60007

    input_tensor_expected = torch.load("tests/baseline/batch_processor/input_forced.pt")
    output_tensor_expected = torch.load("tests/baseline/batch_processor/output_forced.pt")

    assert (input_tensor == input_tensor_expected).all()
    # Target is now only mask channel (channel 0)
    assert (output_tensor == output_tensor_expected[:, 0:1]).all()

    rng = DummyScalarRNG(0.2)
    batch_processor = BurnerBatchProcessor(
        burner=burner,
        dt=30,
        eval=False,
        sampler=ConstSampler(0.3),
        rng=rng
    )
    assert batch_processor.dt == 30

    input_tensor, output_tensor = batch_processor(pred, true, epoch=8, batch_idx=9, t=90)
    assert rng.last_seed_used == 80009

    batch_processor = BurnerBatchProcessor(
        burner=burner,
        dt=30,
        eval=True
    )
    assert batch_processor.dt == 30

    pred = torch.stack([dataset[1], dataset[0]])
    true = torch.stack([dataset[0], dataset[1]])

    input_tensor2, output_tensor2 = batch_processor(pred, true, epoch=10, batch_idx=11, t=90)

    input_tensor_expected = torch.load("tests/baseline/batch_processor/input_autoreg.pt")
    output_tensor_expected = torch.load("tests/baseline/batch_processor/output_autoreg.pt")

    assert (input_tensor == input_tensor_expected).all()
    assert (input_tensor2 == input_tensor_expected).all()
    assert (output_tensor == output_tensor_expected[:, 0:1]).all()
    assert (output_tensor2 == output_tensor_expected[:, 0:1]).all()


def test_trainer(dataloader):
    torch.manual_seed(42)

    dataset = WildfireDataset(dataloader)
    transform = MinMaxPerChannel(dataset.min_val, dataset.max_val)
    dataset = TransformedDataset(dataset, transform)

    burner = ForwardBurnProcess()

    train_batch_processor = BurnerBatchProcessor(
        burner=burner,
        dt=1/48,
        eval=False,
        sampler=ScheduledSampler(k=0.1, t0=40),
        rng=ScalarRNG()
    )
    val_batch_processor = BurnerBatchProcessor(
        burner=burner,
        dt=1/48,
        eval=True
    )

    # share loader for train and val
    loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    def get_trainer(epochs):
        model = MK_UNet_Regression(
            in_channels=14,
            out_channels=1,
            channels=[16, 32, 64, 96, 160],
            final_activation='sigmoid'
        )

        checkpoint_cb = ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            filepath='./checkpoints_test/best-model-{epoch:02d}-{val_loss:.2f}.pt'
        )

        train_writer = SummaryWriter("training_test/train")
        val_writer = SummaryWriter("training_test/val")

        tensorboard_cb = TensorBoardCallback(
            train_writer=train_writer,
            val_writer=val_writer
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            5e-4,
            weight_decay=1e-4
        )

        trainer = ForwardBurnTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=nn.BCELoss(),
            train_loader=loader,
            val_loader=loader,
            train_batch_processor = train_batch_processor,
            val_batch_processor = val_batch_processor,
            callbacks=[checkpoint_cb, tensorboard_cb],
            epochs=epochs,
            max_t=3/48
        )

        return trainer

    shutil.rmtree('./training_test', ignore_errors=True)

    trainer = get_trainer(epochs=5)

    eval_before = trainer.evaluate()
    assert isinstance(eval_before['val_loss'], float)

    shutil.rmtree('./checkpoints_test', ignore_errors=True)

    trainer.fit()

    eval_after = trainer.evaluate()
    assert eval_after['val_loss'] < eval_before['val_loss']

    folder = Path('./checkpoints_test')
    pattern = re.compile(r"best-model-\d{2}-\d+\.\d{2}\.pt")
    matching_files = [
        p for p in folder.iterdir() 
        if p.is_file() and pattern.fullmatch(p.name)
    ]
    assert matching_files

    last_checkpoint = max(matching_files, key=lambda p: p.name)

    trainer = get_trainer(epochs=10)
    trainer.load_checkpoint(last_checkpoint)

    eval_before_resumed = trainer.evaluate()
    assert eval_before_resumed['val_loss'] <= eval_after['val_loss']

    trainer.fit()

    eval_after_resumed = trainer.evaluate()
    assert eval_after_resumed['val_loss'] < eval_after['val_loss']

    train_acc = EventAccumulator("training_test/train")
    train_acc.Reload()
    assert len(set(s.step for s in train_acc.Scalars("Loss"))) == 10

    val_acc = EventAccumulator("training_test/val")
    val_acc.Reload()
    assert len(set(s.step for s in val_acc.Scalars("Loss"))) == 10


