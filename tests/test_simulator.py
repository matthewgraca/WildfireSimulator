import torch
import pytest
import copy
import numpy as np

from wildfire_simulator.simulators import ForwardBurnSimulator, fire_burn_step
from wildfire_simulator.transforms import MinMaxPerChannel

def test_burn_step():
    inputs = torch.zeros((1, 13, 500, 500))
    inputs_copy = copy.deepcopy(inputs)
    def model(data):
        chan1 = torch.full((1, 1, 512, 512), 1.0)
        chan2 = torch.full((1, 1, 512, 512), 2.0)
        return [torch.cat((chan1, chan2), dim=1)]
    outputs = fire_burn_step(0, model, inputs)
    assert (inputs == inputs_copy).all()
    assert outputs.shape == (1, 13, 500, 500)
    assert (outputs[0][0] == 1.0).all()
    assert (outputs[0][1] == 2.0).all()


def test_simulator():
    def model(data):
        return data * 2

    def step(t, model, data):
        return model(t + data)

    class FakeTransform:
        def __call__(self, data):
            return self.transform(data)
        def transform(self, data):
            return data / 5
        # the inverse transform is defined incorrectly so that it isn't transparent
        def inverse(self, data):
            return data * 3

    transform = FakeTransform()

    simulator = ForwardBurnSimulator(
        t0=4,
        data=10,
        model=model,
        step=step,
        transform=transform,
        dt=2,
        max_t=20
    )

    assert simulator.run_to(14) == pytest.approx(244.8)
    assert simulator.run_to(14, return_history=True) == pytest.approx([10, 13.2, 28.2, 58.8, 120.6, 244.8])


def test_simulator_integration():
    inputs = torch.zeros((1, 13, 500, 500))

    def model(data):
        return torch.full((1, 2, 512, 512), 2.0)

    transform = MinMaxPerChannel(np.full((13,), 0.0), np.full((13,), 2.0))

    simulator = ForwardBurnSimulator(
        data=inputs,
        model=model,
        step=fire_burn_step,
        transform=transform,
        dt=1,
        max_t=2
    )

    output = simulator.run_to(2)

    # Channel 0 (mask) is thresholded: model outputs 2.0 > 0.5 → 1.0, inverse transformed → 2.0
    # Channel 1 (arrival) passes through: model outputs 2.0, inverse transformed → 4.0
    assert (output[:, 0:1] == 2).all()
    assert (output[:, 1:2] == 4).all()

