import os
import random

def combine_datasets(dir1, dir2):

    with open(os.path.join(dir1, "train.txt"), "r") as f:
        train1 = f.readlines()
    with open(os.path.join(dir1, "val.txt"), "r") as f:
        val1 = f.readlines()
    
    with open(os.path.join(dir2, "train.txt"), "r") as f:
        train2 = f.readlines()
    with open(os.path.join(dir2, "val.txt"), "r") as f:
        val2 = f.readlines()
    
    all_lines = train1 + train2 + val1 + val2
    random.shuffle(all_lines)

    with open("preprocessed_data/val.txt", "w") as f:
        f.writelines(all_lines[:len(val1)])
        
    with open("preprocessed_data/train.txt", "w") as f:
        f.writelines(all_lines[len(val1):])

    print("Done.")

if __name__ == "__main__":
    preprocessed_dir1 = "preprocessed_data/LJSpeech"
    preprocessed_dir2 = "preprocessed_data/old"
    combine_datasets(preprocessed_dir1, preprocessed_dir2)

    pass
