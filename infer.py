from keras.models import load_model
import utils
import os
from glob import glob
import argparse

parser = argparse.ArgumentParser(description='transformer net inference script')
parser.add_argument('--test_dir', required=True, type=str, help="test images path")
parser.add_argument('--model_path', required=True, type=str, help="trained model path")

args = parser.parse_args()


if __name__ == "__main__":
    test_path = glob(os.path.join(args.test_dir, "*"))
    save_dir = "./result"
    utils.make_dir(save_dir)

    model = load_model(args.model_path)

    for path in test_path:
        filename = os.path.basename(path)
        input_img = utils.img_load(path) / 255.
        infer_img = model(input_img)[0]
        utils.img_save(infer_img.numpy() * 255., os.path.join(save_dir, filename))
