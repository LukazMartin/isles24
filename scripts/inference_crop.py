"""
Save logits and predictions using cropped sliding window inference.
"""

from pathlib import Path
import json
from tqdm import tqdm

from isles.swin.config import SwinTrainConfig
from isles.swin.transforms import get_val_transforms
from isles.swin.training import get_dataloader
from isles.swin.evaluation import final_evaluation


def main():

    # Parameter override for inference
    roi_size = 64
    overlap = 0.5
    inferer_crop_margin = 10

    data_root = Path("/home/renku/work/data-local")
    run_ids = ["run-021", "run-022"]

    for run_id in tqdm(run_ids, "Evaluating runs"):
        run_dir = data_root / f"runs/{run_id}"
        sweep_dir = run_dir / "inference-crop"
        checkpoint_path = run_dir / "checkpoints/best_model.pt"

        config = SwinTrainConfig.from_json(run_dir / "config.json")
        with open(run_dir / "datalist.json") as file:
            datalist = json.load(file)

        val_loader = get_dataloader(
            datalist=datalist,
            key="validation",
            transforms=get_val_transforms(config),
            batch_size=config.batch_size,
            cache_rate=0.0,
        )

        out_dir = (
            sweep_dir / f"roi_{roi_size}-overlap_{overlap}-crop_{inferer_crop_margin}"
        )
        final_evaluation(
            checkpoint_path=checkpoint_path,
            val_loader=val_loader,
            config=config,
            out_dir=out_dir,
            save_logits=True,
            roi_size=roi_size,
            val_overlap_final=overlap,
            inferer_crop_margin=inferer_crop_margin,
        )
        params = {
            "roi_size": roi_size,
            "val_overlap_final": overlap,
            "inferer_crop_margin": inferer_crop_margin,
        }
        with open(out_dir / "params.json", "w") as file:
            json.dump(params, file, indent=2)


if __name__ == "__main__":
    main()
