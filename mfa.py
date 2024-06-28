import os
import shutil

def create_mfa_directory(base_dir):

    dirs = [name for name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, name))]

    for dir in dirs:
        # os.makedirs(os.path.join(new_dir, dir), exist_ok=True)
        for filename in os.listdir(os.path.join(base_dir, dir)):
            if filename.endswith("_src.lab"):
                new_filename = filename.replace("_src.lab", ".lab")
                shutil.copy(os.path.join(base_dir, dir, filename), os.path.join(base_dir, dir, new_filename))
                
                
if __name__ == "__main__":
    dir = "raw_data/Spanish"
    new_dir = "raw_data/MFA"
    create_mfa_directory(dir)

