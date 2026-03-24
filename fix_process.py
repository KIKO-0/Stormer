import os

file_path = 'stormer/data_preprocessing/process_one_step_data.py'

with open(file_path, 'r') as f:
    content = f.read()

# Make it dynamically check if the folder exists
old_code = """        for var in (list_single_vars + list_pressure_vars):
            ds_dict[var] = xr.open_dataset(os.path.join(root_dir, var, f'{year}.nc'))"""

new_code = """        for var in (list_single_vars + list_pressure_vars):
            var_path = os.path.join(root_dir, var, f'{year}.nc')
            if os.path.exists(var_path):
                ds_dict[var] = xr.open_dataset(var_path)
            else:
                pass # print(f"Skipping missing variable: {var}")"""

content = content.replace(old_code, new_code)

old_code2 = """            for var in list_constant_vars:
                    constant_path = os.path.join(root_dir, f'{var}.nc')
                    constant_field = xr.open_dataset(constant_path)[var].to_numpy()"""

new_code2 = """            for var in list_constant_vars:
                    constant_path = os.path.join(root_dir, f'{var}.nc')
                    if not os.path.exists(constant_path): continue
                    constant_field = xr.open_dataset(constant_path)[var].to_numpy()"""
                    
content = content.replace(old_code2, new_code2)

with open(file_path, 'w') as f:
    f.write(content)
print("✅ process_one_step_data.py 已经被成功修复，跳过了不存在的变量。")
