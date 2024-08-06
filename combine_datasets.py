import os
import random
import argparse

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

def get_directory_duration(directory):

    seconds = 0
    for speaker in os.listdir(os.path.join(directory, 'TextGrid')):
        if os.path.isdir(os.path.join(directory, 'TextGrid', speaker)):
            for file in os.listdir(os.path.join(directory, 'TextGrid', speaker)):
                with open(os.path.join(directory, 'TextGrid', speaker, file), 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if 'xmax' in line:
                            seconds += float(line.strip().split(' ')[-1])
                            break
    
    print(f"Total duration is : ", seconds/3600, " hours")

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir1", type=str, default="preprocessed_data/LJSpeech")
    parser.add_argument("--dir2", type=str, default="preprocessed_data/old")
    args = parser.parse_args()

    # combine_datasets(args.dir1, args.dir2)

    get_directory_duration(args.dir1)
    get_directory_duration(args.dir2)

    pass
