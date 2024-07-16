import argparse
import yaml
from preprocessor import miipher


def main(config, speakers):
    audio_processor = miipher.MiipherInference(config)
    audio_processor.process_directory(speakers)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to preprocess.yaml")
    parser.add_argument("--speakers", type=str, 
                        help="comma separated speakers to preprocess, e.g. 'Augmented,Bryant' if none given will preprocess all speakers.")
    args = parser.parse_args()
    
    speakers = args.speakers.split(",") if args.speakers is not None else None
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    main(config, speakers)

    print("Done.")