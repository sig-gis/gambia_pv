import geopandas as gpd
import rasterio
from rasterio.windows import Window
from rasterio.features import rasterize
from shapely.geometry import box
from sklearn.model_selection import train_test_split
import os
import numpy as np


def group_geometries_within_distance(gdf: gpd.GeoDataFrame, distance: float) -> gpd.GeoDataFrame:
    crs = gdf.crs
    gdf_proj = gdf.to_crs(epsg=3857).copy()
    gdf_proj.geometry = gdf_proj.geometry.buffer(distance)
    dissolved = gdf_proj.dissolve()
    exploded = dissolved.explode().reset_index(drop=True)
    result = exploded.to_crs(crs)
    print(len(gdf), len(result))
    return result


def extract_and_save_samples(gdf, gdf_subset, raster, output_dir, suffix, start_idx=0, window_size=512):
    half_size = window_size // 2
    sample_id = start_idx
    for _, row in gdf_subset.iterrows():
        centroid = row.geometry.centroid
        x, y = centroid.xy
        row_pix, col_pix = raster.index(x, y)

        col_off = col_pix - half_size
        row_off = row_pix - half_size

        if col_off < 0:
            col_off = 0
        elif col_off + window_size > raster.width:
            col_off = raster.width - window_size

        if row_off < 0:
            row_off = 0
        elif row_off + window_size > raster.height:
            row_off = raster.height - window_size

        window = Window(col_off, row_off, window_size, window_size)

        mask = rasterize(
            gdf.geometry,
            out_shape=(window_size, window_size),
            transform=raster.window_transform(window),
            fill=0,
            default_value=1
        )     
        
        # confirm centroid is within the window
        if not (col_off <= col_pix < col_off + window_size and row_off <= row_pix < row_off + window_size):
            print(f"skipping {col_off} {row_off} {col_pix} {row_pix}")
            continue
        if np.sum(mask) == 0 and suffix != "bg":
            print(f"skipping {sample_id} zero mask")    
            continue

        data = raster.read(window=window)

        out_meta = raster.meta.copy()
        out_meta.update({
            "height": 512,
            "width": 512,
            "transform": raster.window_transform(window)
        })
        out_path = os.path.join(output_dir, "images", f"{sample_id:04d}_{suffix}.tif")
        with rasterio.open(out_path, "w", **out_meta) as dest:
            dest.write(data)
    
        mask_out_path = os.path.join(output_dir, "masks", f"{sample_id:04d}_{suffix}.tif")
        mask_meta = out_meta.copy()
        mask_meta.update({
            "count": 1, 
        })
        with rasterio.open(mask_out_path, "w", **mask_meta) as dest:
            dest.write(mask, 1)

        sample_id += 1

if __name__ == "__main__":
    base_path = os.getenv("BASE_PATH")
    data_path = os.path.join(base_path, "data")
    
    raster = rasterio.open(os.path.join(data_path, "pleides_RGBNED.tif"))
    
    pv = gpd.read_file(os.path.join(data_path, "pv_polygons.geojson")).to_crs(raster.crs)
    bg = gpd.read_file(os.path.join(data_path, "background.geojson")).to_crs(raster.crs)
    
    pvg = group_geometries_within_distance(pv, distance=4)

    train_pv, test_pv = train_test_split(pvg, test_size=0.2, random_state=42)
    train_bg, test_bg = train_test_split(bg, test_size=0.2, random_state=42)

    train_dir = os.path.join(data_path, "train_samples")
    test_dir = os.path.join(data_path, "test_samples")
    
    os.makedirs(os.path.join(train_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(train_dir, "masks"), exist_ok=True)

    os.makedirs(os.path.join(test_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "masks"), exist_ok=True)

    extract_and_save_samples(pv, train_pv, raster, train_dir, "pv", start_idx=0)
    extract_and_save_samples(pv, test_pv, raster, test_dir, "pv", start_idx=0)

    extract_and_save_samples(pv, train_bg, raster, train_dir, "bg", start_idx=len(train_pv))
    extract_and_save_samples(pv, test_bg, raster, test_dir, "bg", start_idx=len(test_pv))
