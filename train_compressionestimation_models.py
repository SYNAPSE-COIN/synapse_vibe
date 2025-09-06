# Script to train the compression prediction model (img_compression_estimation_models.py)
# using a dataset of compressed images (gen_txt2img_imgs2instruct.py)
# --------------------------------------------------------------------------------

import argparse
import json

from rich.panel import Panel
from rich.progress import Progress

import const
from const import _console
from src_models.img_compression_estimation_models import FractalRhythmicCompressor, \
    NormalizedCompressionDataset

def load_json(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        return json.load(f)

def main(args):
    _console.print(Panel.fit("Fractal-Rhythm Compression Model Trainer", style="bold magenta"))

    compressor = FractalRhythmicCompressor(
        initial_hidden_size=args.initial_hidden_size,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        checkpoint_dir=const.dir_checkpoints
    )

    if not args.no_load:
        compressor.load_checkpoint()

    dataset_json = const.dir_image_datasets / f"{args.input_dataset}.json"
    image_dataset = load_json(dataset_json)
    _console.print(f"[green]Loaded dataset with {len(image_dataset['images'])} samples[/green]")

    dataset = NormalizedCompressionDataset(
        image_dataset,
        args.formats,
        (args.min_quality, args.max_quality),
        (args.min_width, args.max_width)
    )

    _console.print(f"[blue]Training with {args.samples_per_epoch} samples/epoch, batch size {args.batch_size}[/blue]")

    with Progress() as progress:
        task = progress.add_task("[cyan]Training", total=args.epochs)

        cached_console = const._console
        const._console = progress.console

        compressor.train_model(
            dataset=dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            samples_per_epoch=args.samples_per_epoch
        )

        progress.update(task, advance=1)
        const._console = cached_console

    _console.print("[green]Training finished. Saving checkpoint.[/green]")
    compressor.save_checkpoint()
    compressor.plot_performance()
    _console.print("[green]Performance chart saved.[/green]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train image compression estimation models")
    parser.add_argument('--input-dataset', type=str, required=True, help="Input dataset name (without .json)")
    parser.add_argument('--initial-hidden-size', type=int, default=10, help="Initial hidden layer size")
    parser.add_argument('--learning-rate', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--epochs', type=int, default=50, help="Number of epochs")
    parser.add_argument('--samples-per-epoch', type=int, default=1000, help="Samples per epoch")
    parser.add_argument('--min-quality', type=int, default=2, help="Minimum quality")
    parser.add_argument('--max-quality', type=int, default=100, help="Maximum quality")
    parser.add_argument('--min-width', type=int, default=64, help="Minimum width")
    parser.add_argument('--formats', nargs='+', default=['webp','jpeg'], help="Compression formats")
    parser.add_argument('--max-width', type=int, default=2048, help="Maximum width")
    parser.add_argument('--save-interval', type=int, default=5, help="Checkpoint save interval (epochs)")
    parser.add_argument('--no-load', action='store_true', help="Skip loading existing checkpoint")
    parser.add_argument('--save-samples', type=str, help="Optional directory to save compressed samples")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'rmsprop'], help="Optimizer")

    args = parser.parse_args()
    main(args)
