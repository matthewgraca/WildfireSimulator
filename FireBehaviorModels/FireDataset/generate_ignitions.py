#!/usr/bin/env python3
"""
generate_ignitions.py

Generate N random ignition point shapefiles from a landscape GeoTIFF.

Ignition points are:
  - Placed on burnable fuel cells (avoids non-burnable: 91, 92, 93, 98, 99)
  - Kept at least `margin` cells away from the landscape edge so the ML
    dataloader's center crop (2*half) stays fully in-bounds
  - Written in the landscape's CRS as Point shapefiles matching the existing
    format (fields: ENTITY, VALUE; one point per file)

Output files:  <outdir>/ignition_0.shp ... ignition_<N-1>.shp

Usage:
  python generate_ignitions.py --landscape angeles.tif --n 50 --outdir Ignitions
"""

import os
import argparse
import numpy as np
import rioxarray
import geopandas as gpd
from shapely.geometry import Point
from rasterio.transform import xy

# Non-burnable Scott & Burgan FBFM40 fuel codes
NONBURNABLE = {91, 92, 93, 98, 99}


def parse_args():
    p = argparse.ArgumentParser(description="Generate ignition point shapefiles from a landscape TIF.")
    p.add_argument("--landscape-name", help="Landscape name; resolves to ./landscapes/<name>/<name>.tif and writes to ./landscapes/<name>/Ignitions. Overridden by --landscape/--outdir if given.")
    p.add_argument("--landscape", help="Explicit path to landscape GeoTIFF (fuel model expected on band 4, index 3).")
    p.add_argument("--n", type=int, default=50, help="Number of ignition points to generate.")
    p.add_argument("--outdir", help="Output directory for ignition_*.shp files.")
    p.add_argument("--fuel-band", type=int, default=3, help="Zero-based band index of the fuel model layer (default 3).")
    p.add_argument("--margin", type=int, default=250, help="Minimum distance in cells from the landscape edge (default 250 to match ML crop half-size).")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    p.add_argument("--nodata", type=float, default=-9999, help="NoData value to treat as non-burnable (default -9999).")
    args = p.parse_args()

    # Resolve paths from --landscape-name if the explicit options weren't given
    if args.landscape_name:
        base = os.path.join("landscapes", args.landscape_name)
        if not args.landscape:
            args.landscape = os.path.join(base, f"{args.landscape_name}.tif")
        if not args.outdir:
            args.outdir = os.path.join(base, "Ignitions")

    if not args.landscape:
        p.error("provide either --landscape-name or --landscape")
    if not args.outdir:
        args.outdir = "Ignitions"

    return args


def main():
    args = parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    da = rioxarray.open_rasterio(args.landscape)
    crs = da.rio.crs
    transform = da.rio.transform()

    fuel = da.isel(band=args.fuel_band).values
    n_rows, n_cols = fuel.shape

    # Build a boolean mask of valid ignition cells:
    #   burnable fuel AND inside the edge margin
    burnable = np.isin(fuel, list(NONBURNABLE), invert=True)
    burnable &= (fuel != args.nodata)

    edge_ok = np.zeros_like(burnable, dtype=bool)
    r0, r1 = args.margin, n_rows - args.margin
    c0, c1 = args.margin, n_cols - args.margin
    if r0 >= r1 or c0 >= c1:
        raise ValueError(
            f"Margin {args.margin} too large for landscape of size {n_rows}x{n_cols}. "
            f"Reduce --margin."
        )
    edge_ok[r0:r1, c0:c1] = True

    valid = burnable & edge_ok
    valid_rows, valid_cols = np.where(valid)
    n_valid = len(valid_rows)

    if n_valid == 0:
        raise RuntimeError("No valid burnable cells found within the edge margin.")
    if n_valid < args.n:
        raise RuntimeError(f"Only {n_valid} valid cells available, but {args.n} requested.")

    # Sample N unique cells without replacement
    chosen = np.random.choice(n_valid, size=args.n, replace=False)

    os.makedirs(args.outdir, exist_ok=True)

    for i, idx in enumerate(chosen):
        row = int(valid_rows[idx])
        col = int(valid_cols[idx])
        # Convert (row, col) to projected x/y at the cell center
        x, y = xy(transform, row, col, offset="center")
        gdf = gpd.GeoDataFrame(
            {"ENTITY": [0.0], "VALUE": [0.0]},
            geometry=[Point(x, y)],
            crs=crs,
        )
        out_path = os.path.join(args.outdir, f"ignition_{i}.shp")
        gdf.to_file(out_path)
        print(f"[{i+1}/{args.n}] ignition_{i}.shp  fuel={int(fuel[row, col])}  (row={row}, col={col})  ->  ({x:.1f}, {y:.1f})")

    print(f"\nDone. Wrote {args.n} ignition shapefiles to '{args.outdir}'.")


if __name__ == "__main__":
    main()
