import os
import shutil

def create_mfa_directory(base_dir):

    dirs = [name for name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, name))]

    for dir in dirs:
        # os.makedirs(os.path.join(new_dir, dir), exist_ok=True)
        for filename in os.listdir(os.path.join(base_dir, dir)):
            if filename.endswith("_src.lab"):
                new_filename = filename.replace("_src.lab", ".lab")
                os.rename(os.path.join(base_dir, dir, filename), os.path.join(base_dir, dir, new_filename))                

def get_oov_words(oov_log_file, dict, oov_words):
    import json
    import re

    with open(oov_log_file, "r") as f:
        lines = f.readlines()

    my_line = lines[2].replace('(', '"(').replace(')', ')"')

    # Step 1: Temporarily replace single quotes within parentheses with a placeholder
    def replace_with_placeholder(match):
        return match.group(0).replace("'", "<PLACEHOLDER>")
    
    temp_replaced = re.sub(r'\([^)]*\)', replace_with_placeholder, my_line)

    # Step 2: Replace all single quotes with double quotes
    almost_fixed_json = temp_replaced.replace("'", '"')

    # Step 3: Restore placeholders back to single quotes
    fixed_json = almost_fixed_json.replace("<PLACEHOLDER>", "'")

    # New Step: Enclose text within angle brackets in single quotes
    fixed_json = re.sub(r'<([^>]*)>', r'"<\1>"', fixed_json)

    data = json.loads(fixed_json)

    with open(dict, "r") as f:
        lines = f.readlines()
    dict_words = [line.strip().split('\t')[0] for line in lines]

    print(dict_words[:10])
    
    with open(oov_words, "w") as f:
        for val in data.values():
            if val['word'] not in dict_words:
                f.write(val['word'] + '\n')

    print(bool('<unk>' in dict_words), bool('aang' in dict_words))

    return


def merge_dictionaries(pretrained_dict_path, generated_dict_path):
    import re

    # Read the pretrained dictionary
    with open(pretrained_dict_path, 'r') as f:
        lines = f.readlines()

    prob1, prob2, prob3, prob4 = [], [], [], []
    for i in range(len(lines)):
        parts = re.split(r'\s+', lines[i].strip())
        word = parts[0]
        prob1.append(float(parts[1]))
        prob2.append(float(parts[2]))
        prob3.append(float(parts[3]))
        prob4.append(float(parts[4]))

    prob1 = round(sum(prob1) / len(prob1), 2)
    prob2 = round(sum(prob2) / len(prob2), 2)
    prob3 = round(sum(prob3) / len(prob3), 2)
    prob4 = round(sum(prob4) / len(prob4), 2)

    # Read the generated dictionary
    with open(generated_dict_path, 'r') as f:
        lines = f.readlines()
        
    for i in range(len(lines)):
        parts = re.split(r'\s+', lines[i].strip())
        word = parts[0]
        pronunciation = ' '.join(parts[1:])
        lines[i] = f"{word}\t{prob1}\t{prob2}\t{prob3}\t{prob4}\t{pronunciation}\n"

    # Write to the pretrained dictionary
    with open(pretrained_dict_path, 'a') as f:
        f.writelines(lines)




if __name__ == "__main__":
    dir = "raw_data/LJSpeech"
    # create_mfa_directory(dir)

    oov_file = "/home/ditto/Documents/MFA/Spanish_mfa/Spanish_mfa/split3/log/normalize_oov.log"
    dict = "/home/ditto/Documents/MFA/pretrained_models/dictionary/spanish_mfa.dict"
    oov_words = "/home/ditto/Documents/MFA/Spanish_mfa/oov_words.txt"
    # get_oov_words(oov_file, dict, out_file)

    pretrained_dict_path = '/home/ditto/Documents/MFA/pretrained_models/dictionary/spanish_mfa.dict'
    generated_dict_path = '/home/ditto/Documents/MFA/Spanish_mfa/dictionary/generated_dictionary.txt'

    pretrained_dict_path = '/home/ditto/Documents/MFA/pretrained_models/dictionary/spanish_mfa.dict'
    generated_dict_path = '/home/ditto/Ditto/FastSpeech2/lexicon/librispeech-lexicon.txt'
    merge_dictionaries(pretrained_dict_path, generated_dict_path)

    pass
