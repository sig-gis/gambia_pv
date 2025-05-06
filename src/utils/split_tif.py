import rasterio
import os
import rasterio


base_path = "/home/ceoas/truongmy/emapr/gambia_pv"
all_path = os.path.join(base_path, "pleides_RGBNED.tif")

with rasterio.open(all_path) as src:
    count = src.count
    bands = "RGBNED"
    profile = src.profile
    profile.update(count=1)
    for i in range(count):
        band = src.read(i+1)
        with rasterio.open(os.path.join(base_path, f"pleides_{bands[i]}.tif"), "w", **profile) as dst:
            dst.write(band, 1)
