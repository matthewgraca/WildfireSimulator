import numpy as np
import torch
from torch.utils.data import Dataset

class WildfireDataset(Dataset):
    def __init__(self, dataloader):
        self.loader = dataloader

        # Compute channel-wise min and max across all trials/frames for normalization
        n_channels = 13
        min_val = np.full(n_channels,  np.inf)
        max_val = np.full(n_channels, -np.inf)
        for i in range(self.__len__()):
            arr = self._load_raw_channels(i)
            frame_min = np.min(arr, axis=(1, 2))
            frame_max = np.max(arr, axis=(1, 2))
            min_val = np.minimum(min_val, frame_min)
            max_val = np.maximum(max_val, frame_max)
        self.min_val = min_val
        self.max_val = max_val

    def _load_raw_channels(self, idx):
        trial = self.loader.trials[idx]
        ig_idx = trial["ignition"]
        cy, cx = self.loader.ignitions[ig_idx]
        half = 250

        # 8 landscape layers in order
        land_layers = [
            self.loader.elevation,
            self.loader.slope,
            self.loader.aspect,
            self.loader.fuel,
            self.loader.canopy_cover,
            self.loader.stand_height,
            self.loader.canopy_base_height,
            self.loader.canopy_bulk_density
        ]
        crops = [
            np.asarray(arr[cy-half:cy+half, cx-half:cx+half], dtype=np.float32)
            for arr in land_layers
        ]

        # 2 fire channels (mask and arrival time)
        fire_mask = np.asarray(
            trial["fire"][0][cy-half:cy+half, cx-half:cx+half],
            dtype=np.float32,
        )
        fire_arr = np.asarray(
            trial["fire"][1][cy-half:cy+half, cx-half:cx+half],
            dtype=np.float32,
        )

        # scalar channels broadcast to 500×500
        # Convert wind speed + direction (degrees) to U/V cartesian components
        # Meteorological convention: direction is where wind comes FROM, clockwise from north
        # U = east-west component (positive = wind blowing east)
        # V = north-south component (positive = wind blowing north)
        wind_speed = trial["windspeed"]
        wind_dir_rad = np.radians(trial["winddir"])
        wind_u = -wind_speed * np.sin(wind_dir_rad)  # east-west
        wind_v = -wind_speed * np.cos(wind_dir_rad)  # north-south

        wu = np.full((500, 500), wind_u, dtype=np.float32)
        wv = np.full((500, 500), wind_v, dtype=np.float32)
        fm = np.full((500, 500), trial["foliar_moisture"], dtype=np.float32)
        # stack all 13 channels
        stacked = np.stack(
            [fire_mask, fire_arr, *crops, wu, wv, fm], axis=0
        )

        return stacked

    def __len__(self):
        return len(self.loader.trials)

    def __getitem__(self, idx):
        channels = self._load_raw_channels(idx)
        return torch.from_numpy(channels).to(torch.float32)


class TransformedDataset:
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.transform(self.dataset[idx])

