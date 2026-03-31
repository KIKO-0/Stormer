import xarray as xr
import os
from tqdm import tqdm

save_dir = "/home/stormer_data/wb2_nc"
file = "1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr"
years = [2018, 2019, 2020]

os.makedirs(save_dir, exist_ok=True)
print("Connecting to GCS...")
ds = xr.open_zarr('gs://weatherbench2/datasets/era5/' + file)
variables = list(ds.keys())
print(f"Found {len(variables)} variables, downloading years: {years}")

for var in tqdm(variables, desc="variables"):
    ds_var = ds[[var]]
    if len(ds_var.dims) < 3:
        out_path = os.path.join(save_dir, f'{var}.nc')
        if os.path.exists(out_path):
            tqdm.write(f"  Skip {var} (exists)")
            continue
        tqdm.write(f"  Saving {var}...")
        try:
            ds_var.to_netcdf(out_path)
        except Exception as e:
            tqdm.write(f"  [ERROR] {var}: {e}")
    else:
        save_dir_var = os.path.join(save_dir, var)
        os.makedirs(save_dir_var, exist_ok=True)
        for year in years:
            out_path = os.path.join(save_dir_var, f'{year}.nc')
            if os.path.exists(out_path):
                tqdm.write(f"  Skip {var}/{year} (exists)")
                continue
            tqdm.write(f"  Downloading {var}/{year}...")
            try:
                ds_var.sel(time=str(year)).to_netcdf(out_path)
                tqdm.write(f"  Done {var}/{year}")
            except Exception as e:
                tqdm.write(f"  [ERROR] {var}/{year}: {e}")
                if os.path.exists(out_path):
                    os.remove(out_path)

print("All done!")
