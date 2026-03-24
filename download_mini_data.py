import argparse
import xarray as xr
import os
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr", help="Zarr file from WB2")
    parser.add_argument("--save_dir", type=str, default="./mini_wb2_nc", help="Directory to save the mini dataset")
    parser.add_argument("--year", type=int, default=2020, help="Year to download")
    args = parser.parse_args()
    
    file = args.file
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Connecting to Google Cloud Storage: gs://weatherbench2/datasets/era5/{file} ...")
    # You might need gcsfs installed: pip install gcsfs zarr xarray
    try:
        ds = xr.open_zarr('gs://weatherbench2/datasets/era5/' + file)
    except Exception as e:
        print("Failed to access GCS.", e)
        print("Please ensure you have run: pip install gcsfs zarr xarray dask netCDF4")
        return
        
    years = [args.year]
    # We only want the variables that are defined in process_one_step_data.py (or all of them)
    variables = list(ds.keys())
    
    print(f"Downloading a tiny subset (2 days of Jan {years[0]}) for all variables...")
    
    for var in tqdm(variables, desc="variables", position=0):
        ds_var = ds[[var]]
        if len(ds_var.dims) < 3: # constant variables
            # just save it
            ds_var.to_netcdf(os.path.join(save_dir, f'{var}.nc'))
        else:
            save_dir_var = os.path.join(save_dir, var)
            os.makedirs(save_dir_var, exist_ok=True)
            for year in years:
                # ONLY GET JAN 1 TO JAN 2 ! (2 days = 8 timesteps if 6-hourly)
                # This drastically reduces the size from ~1.5GB per variable/year to a few MBs
                ds_var_year = ds_var.sel(time=slice(f'{year}-01-01', f'{year}-01-02'))
                ds_var_year.to_netcdf(os.path.join(save_dir_var, f'{year}.nc'))
                
    print("\n✅ Mini dataset download complete!")
    print(f"Saved to: {save_dir}")
    print("Now you can run:")
    print(f"python stormer/data_preprocessing/process_one_step_data.py --root_dir {save_dir} --save_dir ./mini_wb2_h5df --start_year {years[0]} --end_year {years[0]+1} --split train")

if __name__ == "__main__":
    main()
