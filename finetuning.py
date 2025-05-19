import argparse

from config.main import get_config
from utils.get_model import get_model
from utils.fine_tuning import finetuning

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model to load: vit, swin, convnext, densenet, efficientnet, customcnn, resnet18, vgg16",
    )
    args = parser.parse_args()

    model = get_model(args.model)
    print(f"Loaded model: {args.model}")

    config = get_config(args.model)

    finetuning(config, model)
