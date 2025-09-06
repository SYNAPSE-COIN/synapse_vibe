from pathlib import Path
from rich.console import Console

base_path = Path(__file__).parent

path_instruct = base_path / Path("instruct_datasets")
path_instruct.mkdir(exist_ok=True, parents=True)

path_images = base_path / Path("image_datasets")
path_images.mkdir(exist_ok=True, parents=True)

path_audio = base_path / Path("audio_datasets")
path_audio.mkdir(exist_ok=True, parents=True)

path_models = base_path / Path("checkpoints")
path_models.mkdir(exist_ok=True, parents=True)

path_figures = base_path / Path(".plots")
path_figures.mkdir(exist_ok=True, parents=True)

_console = Console()

import matplotlib as mpl
mpl.use('TKAgg')  # avoiding PyQt dependency which can be default in some setups
