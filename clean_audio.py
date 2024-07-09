import argparse
import yaml
from preprocessor import miipher


def main(config):
    audio_processor = miipher.MiipherInference(config)
    audio_processor.process_directory()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to preprocess.yaml")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    main(config)

    print("Done.")