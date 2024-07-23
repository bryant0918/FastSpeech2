import argparse

import yaml

from preprocessor.preprocessor import Preprocessor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to preprocess.yaml")

    # Don't use this unless you plan on ONLY using those speakers during training.
    # parser.add_argument("--speakers", type=str, 
    #                     help="comma separated speakers to preprocess, e.g. 'Augmented,Bryant' if none given will preprocess all speakers.")
    
    args = parser.parse_args()
    
    speakers = args.speakers.split(",") if args.speakers is not None else None
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    preprocessor = Preprocessor(config)
    preprocessor.build_from_path(speakers)
