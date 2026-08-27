import os
import numpy as np
import rioxarray
import geopandas as gpd
from dotenv import load_dotenv

class TrialFileLoader:
    """Loads data of trial file as a dict"""

    def load(self, file_path):
        trial_ds = rioxarray.open_rasterio(file_path)

        # each trial GeoTIFF has one band: fire arrival time,
        # with NaN where fire never arrives
        arr = trial_ds.isel(band=0).values

        # treat -9999 as nodata (same as NaN)
        arr[arr == -9999] = np.nan

        # mask: 1 where fire arrived (non‑NaN), 0 elsewhere
        mask = (~np.isnan(arr)).astype(np.uint8)

        # replace NaN with 0 so the array can be used numerically
        arrival = np.where(np.isnan(arr), 0, arr)
        stacked = np.stack([mask, arrival], axis=0)

        # parse metadata from filename (example: "trail_I0_WS12_WD312_M74.tif")
        base = os.path.splitext(os.path.basename(file_path))[0]
        parts = base.split('_')

        # extract data from file name
        ign_str = next(p for p in parts if p.startswith('I'))
        ignition = int(ign_str[1:])
        ws_str = next(p for p in parts if p.startswith('WS'))
        windspeed = int(ws_str[2:])
        wd_str = next(p for p in parts if p.startswith('WD'))
        winddir = int(wd_str[2:])
        m_str = next(p for p in parts if p.startswith('M'))
        foliar_moisture = int(m_str[1:])

        # Load per-cell wind grids if available (v2 dataset)
        dir_path = os.path.dirname(file_path)
        windspeed_grid_path = os.path.join(dir_path, base + "_windspeed.tif")
        winddir_grid_path = os.path.join(dir_path, base + "_winddir.tif")

        wind_speed_grid = None
        wind_dir_grid = None
        if os.path.exists(windspeed_grid_path) and os.path.exists(winddir_grid_path):
            ws_ds = rioxarray.open_rasterio(windspeed_grid_path)
            wind_speed_grid = ws_ds.isel(band=0).values.astype(np.float32)
            wind_speed_grid[wind_speed_grid == -9999] = 0

            wd_ds = rioxarray.open_rasterio(winddir_grid_path)
            wind_dir_grid = wd_ds.isel(band=0).values.astype(np.float32)
            wind_dir_grid[wind_dir_grid == -9999] = 0

        return {
            "filename": os.path.basename(file_path),
            "fire": stacked,
            "ignition": ignition,
            "windspeed": windspeed,
            "winddir": winddir,
            "foliar_moisture": foliar_moisture,
            "wind_speed_grid": wind_speed_grid,
            "wind_dir_grid": wind_dir_grid,
        }


class TrialCollection:
    """Fetch all the trial file paths and then fetch the data as needed using the file loader"""

    def __init__(self, loader):
        # Load environment from .env file
        load_dotenv()

        self.loader = loader

        # Load fire trial arrival times from TRIALS directory
        trials_dir = os.getenv("TRIALS")
        self.files = []
        if trials_dir and os.path.isdir(trials_dir):
            for fname in sorted(os.listdir(trials_dir)):
                if fname.lower().endswith('.tif') or fname.lower().endswith('.tiff'):
                    fpath = os.path.join(trials_dir, fname)
                    self.files.append(fpath)

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        return self.loader.load(self.files[idx])


class WildfireDataLoader:
    """Loads landscape and trial GeoTIFFs as well as ignition shape files"""

    def __init__(self, trials):
        # Load environment from .env file
        load_dotenv()

        self.trials = trials

        tif_path = os.getenv("LANDSCAPE")
        if not tif_path:
            raise EnvironmentError(
                "Environment variable 'LANDSCAPE' is not set. "
                "Make sure your .env file contains LANDSCAPE=<path>"
            )

        # Open the GeoTIFF with rioxarray – returns an xarray.DataArray
        da = rioxarray.open_rasterio(tif_path)

        # Store the full DataArray for potential later use
        self._landscape = da

        # The first band (index 0) is elevation
        self.elevation = da.isel(band=0).values
        self.elevation[self.elevation == -9999] = 0

        # The second band (index 1) is slope
        self.slope = da.isel(band=1).values
        self.slope[self.slope == -9999] = 0

        # The third band (index 2) is aspect
        self.aspect = da.isel(band=2).values
        self.aspect[self.aspect == -9999] = 0

        # Band 3: fuel model (FBFM40)
        self.fuel = da.isel(band=3).values
        self.fuel[self.fuel == -9999] = 0

        # Band 4: canopy cover (CC)
        self.canopy_cover = da.isel(band=4).values
        self.canopy_cover[self.canopy_cover == -9999] = 0

        # Band 5: stand height (CH)
        self.stand_height = da.isel(band=5).values
        self.stand_height[self.stand_height == -9999] = 0

        # Band 6: canopy base height (CBH)
        self.canopy_base_height = da.isel(band=6).values
        self.canopy_base_height[self.canopy_base_height == -9999] = 0

        # Band 7: canopy bulk density (CBD)
        self.canopy_bulk_density = da.isel(band=7).values
        self.canopy_bulk_density[self.canopy_bulk_density == -9999] = 0

        # Load ignition points from IGNITIONS directory
        ignitions_dir = os.getenv("IGNITIONS")
        self.ignitions = {}
        if ignitions_dir and os.path.isdir(ignitions_dir):
            from rasterio.transform import rowcol
            # Get transform and CRS from the already opened landscape dataset
            transform = self._landscape.rio.transform()
            landscape_crs = self._landscape.rio.crs
            for fname in sorted(os.listdir(ignitions_dir)):
                if not (fname.lower().endswith('.shp') and fname.startswith('ignition_')):
                    continue
                # Extract ignition number from filename "ignition_<N>.shp"
                try:
                    ign_num = int(fname[len('ignition_'):-4])
                except ValueError:
                    # Skip files with unexpected naming
                    continue
                fpath = os.path.join(ignitions_dir, fname)
                gdf = gpd.read_file(fpath)
                if len(gdf) == 0:
                    continue
                # Use the first geometry (expects a single Point per file)
                geom = gdf.geometry.iloc[0]
                # Reproject to landscape CRS when necessary
                if gdf.crs is not None and landscape_crs is not None and gdf.crs != landscape_crs:
                    gdf = gdf.to_crs(landscape_crs)
                    geom = gdf.geometry.iloc[0]
                # Convert to pixel coordinates (row, col) using the affine transform
                rows, cols = rowcol(transform, [geom.x], [geom.y])
                pixel = (int(rows[0]), int(cols[0]))
                self.ignitions[ign_num] = pixel

