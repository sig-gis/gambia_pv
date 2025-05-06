import rasterio
import numpy as np
import os
import rasterio
from rasterio.merge import merge
import glob

base_path = "/home/ceoas/truongmy/emapr/gambia_pv"
names = ["dimapV2_pneo_PNEO3_acq20240205_del7e7745c6_{}.tif",
         "dimapV2_pneo_PNEO3_acq20240205_del78947a6c_{}.tif",
         "dimapV2_pneo_PNEO3_acq20240205_delfadfa8da_{}.tif",
         "dimapV2_pneo_PNEO4_acq20231228_del5243c677_{}.tif"
]

for name in names:
    print(name)
    tif1_path = os.path.join(base_path, name.format("RGB"))
    tif2_path = os.path.join(base_path, name.format("NED"))
    output_path = os.path.join(base_path, name.format("RGBNED"))

    with rasterio.open(tif1_path) as src1:
        bands1 = src1.read()
        profile = src1.profile

    with rasterio.open(tif2_path) as src2:
        bands2 = src2.read()

    if bands1.shape[1:] != bands2.shape[1:]:
        raise ValueError("Input TIFFs must have the same dimensions")

    merged_bands = np.concatenate([bands1, bands2], axis=0)
    profile.update(count=6)

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(merged_bands)

tif_files = [name.format("RGBNED") for name in names]
src_files_to_mosaic = [rasterio.open(os.path.join(base_path,f)) for f in tif_files]

mosaic, out_transform = merge(src_files_to_mosaic)

out_meta = src_files_to_mosaic[0].meta.copy()
out_meta.update({
    "driver": "GTiff",
    "height": mosaic.shape[1],
    "width": mosaic.shape[2],
    "transform": out_transform
})

with rasterio.open(os.path.join(base_path, "pleides_RGBNED.tif"), "w", **out_meta) as dest:
    dest.write(mosaic)

for src in src_files_to_mosaic:
    src.close()