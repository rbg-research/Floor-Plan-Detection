import os
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

import gdown
import torch


DEFAULT_CHECKPOINT_URL = "https://drive.google.com/uc?id=1gRB7ez1e4H7a9Y09lLqRuna0luZO5VRK"
DEFAULT_CHECKPOINT_NAME = "model_best_val_loss_var.pkl"
DEFAULT_BLENDER_URL = "https://ftp.nluug.nl/pub/graphics/blender/release/Blender2.93/blender-2.93.1-linux-x64.tar.xz"
DEFAULT_BLENDER_DIR = "2.93.1"


def resolve_checkpoint_path():
    env_path = os.environ.get("FLOORPLAN_CHECKPOINT")
    candidates = [
        env_path,
        DEFAULT_CHECKPOINT_NAME,
        str(Path("model") / DEFAULT_CHECKPOINT_NAME),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    searched = ", ".join([c for c in candidates if c])
    raise FileNotFoundError(
        "Missing trained checkpoint. Expected one of: "
        f"{searched}. Run `python -m utils.asset_setup --checkpoint` first."
    )


def load_trained_checkpoint(map_location=None):
    checkpoint_path = resolve_checkpoint_path()
    return torch.load(checkpoint_path, map_location=map_location)


def ensure_checkpoint(output=DEFAULT_CHECKPOINT_NAME, url=DEFAULT_CHECKPOINT_URL):
    output_path = Path(output)
    if output_path.exists():
        print(f"Checkpoint already exists at {output_path}")
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading checkpoint to {output_path} ...")
    gdown.download(url, str(output_path), quiet=False)
    if not output_path.exists():
        raise FileNotFoundError(f"Checkpoint download did not create {output_path}")
    return output_path


def ensure_blender(
    url=DEFAULT_BLENDER_URL,
    install_dir=DEFAULT_BLENDER_DIR,
    archive_name="blender-2.93.1-linux-x64.tar.xz",
):
    install_path = Path(install_dir)
    blender_bin = install_path / "blender"
    if blender_bin.exists():
        print(f"Blender already exists at {blender_bin}")
        return blender_bin

    archive_path = Path(archive_name)
    print(f"Downloading Blender archive to {archive_path} ...")
    urlretrieve(url, archive_path)

    install_path.mkdir(parents=True, exist_ok=True)
    print(f"Extracting Blender into {install_path} ...")
    with tarfile.open(archive_path, "r:xz") as tar:
        members = tar.getmembers()
        for member in members:
            parts = Path(member.name).parts
            if len(parts) > 1:
                member.name = str(Path(*parts[1:]))
                tar.extract(member, install_path)

    archive_path.unlink(missing_ok=True)
    if not blender_bin.exists():
        raise FileNotFoundError(f"Blender extraction did not create {blender_bin}")
    return blender_bin


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download model checkpoint and optional Blender assets.")
    parser.add_argument("--checkpoint", action="store_true", help="Download the trained checkpoint.")
    parser.add_argument("--blender", action="store_true", help="Download and extract Blender.")
    parser.add_argument("--all", action="store_true", help="Download both checkpoint and Blender.")
    parser.add_argument("--checkpoint-output", default=DEFAULT_CHECKPOINT_NAME, help="Where to save the checkpoint.")
    parser.add_argument("--blender-dir", default=DEFAULT_BLENDER_DIR, help="Where to extract Blender.")
    args = parser.parse_args()

    if not any([args.checkpoint, args.blender, args.all]):
        parser.error("Choose at least one of --checkpoint, --blender, or --all.")

    if args.checkpoint or args.all:
        ensure_checkpoint(output=args.checkpoint_output)
    if args.blender or args.all:
        ensure_blender(install_dir=args.blender_dir)


if __name__ == "__main__":
    main()
