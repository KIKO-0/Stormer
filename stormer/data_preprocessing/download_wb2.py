import argparse
import xarray as xr
import os
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip files that already exist to support resume-like reruns",
    )

    args = parser.parse_args()
    
    file = args.file
    save_dir = args.save_dir
    skip_existing = args.skip_existing
    
    os.makedirs(save_dir, exist_ok=True)
    ds = xr.open_zarr('gs://weatherbench2/datasets/era5/' + file)
    
    years = list(range(1959, 2021+1))
    variables = list(ds.keys())
    
    for var in tqdm(variables, desc="variables", position=0):
        ds_var = ds[[var]]
        if len(ds_var.dims) < 3: # constant variables
            output_path = os.path.join(save_dir, f'{var}.nc')
            if skip_existing and os.path.exists(output_path):
                continue
            ds_var.to_netcdf(output_path)
        else:
            save_dir_var = os.path.join(save_dir, var)
            os.makedirs(save_dir_var, exist_ok=True)
            for year in tqdm(years, desc="years", position=1, leave=False):
                output_path = os.path.join(save_dir_var, f'{year}.nc')
                if skip_existing and os.path.exists(output_path):
                    continue
                ds_var_year = ds_var.sel(time=str(year))
                ds_var_year.to_netcdf(output_path)
            

if __name__ == "__main__":
    main()