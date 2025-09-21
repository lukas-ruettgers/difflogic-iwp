import os

## Invoke this file as `python execution_setup/set_default_directories.py --user MYUSERNAME`
DIR_NAMES = {
    'DATA': 'data',
    'RESULTS': 'results',
    'RESULTS_TEMP': 'results',
    'ANALYSIS': 'analysis'
}

FILENAME = 'directories.py'
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SUPER_DIR = os.path.dirname(THIS_DIR)

if __name__ == "__main__":
    filepath = os.path.join(THIS_DIR, FILENAME)
    
    # Step 1: Read existing variables
    existing_vars = set()
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            for line in f:
                # Strip whitespace and ignore comments or empty lines
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key = line.split('=', 1)[0].strip()
                    existing_vars.add(key)

    # Set default variables if not defined already
    new_lines = []
    for dir_type, dir_name in DIR_NAMES.items():
        var_name = f"DIR_{dir_type}"
        var_definition = f"DIR_{dir_type}='{os.path.join(SUPER_DIR, dir_name)}'\n"
        if var_name not in existing_vars:
            new_lines.append(var_definition)
    
    if new_lines:
        with open(filepath, 'a') as f:
            f.writelines(new_lines)
        print(f"Updated {FILENAME} in folder {THIS_DIR}")
    else:
        print(f"{FILENAME} already contains all required variables.")