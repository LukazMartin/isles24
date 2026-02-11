"""
"Manual" test-time augmentation by aggregating logits
"""

from pathlib import Path
import json
import re
from tqdm import tqdm
import pandas as pd
import nibabel as nib
import numpy as np
from scipy.special import softmax
from isles.metrics import (
    compute_dice_f1_instance_difference,
    compute_absolute_volume_difference,
)


def select_dir(path: Path, selected_params: dict) -> bool:
    """Select or discard directory based on parameters"""
    with open(path / "params.json") as file:
        params = json.load(file)

    roi = params["roi_size"] in selected_params["roi_size"]
    overlap = params["val_overlap_final"] in selected_params["val_overlap_final"]
    blend = params["inferer_blend_mode"] in selected_params["inferer_blend_mode"]
    return all([roi, overlap, blend])


def get_case(name: str) -> str:
    """Get case ID from image name"""
    return re.search(r"sub-stroke\d+", name).group(0)


def main():

    selected_params = {
        "val_overlap_final": [0.2, 0.5],
        "roi_size": [64, 96],
        "inferer_blend_mode": ["constant"],
    }

    sweep_root = Path("data-local/runs/run-016/logit-sweep")
    mask_root = Path("data-local/train/derivatives")
    
    dst_dir = sweep_root / "aggregated"
    dst_dir.mkdir(exist_ok=True, parents=True)
    with open(dst_dir / "params.json", "w") as file:
        json.dump(selected_params, file)

    sweep_dirs = [path for path in sweep_root.glob("*/") if select_dir(path, selected_params)]
    case_list = sorted(get_case(i.name) for i in sweep_dirs[0].glob("logits/*nii.gz"))

    results = []
    for case in tqdm(case_list, "Processing cases"):
        logit_name = f"logits/{case}_ses-01_space-ncct_cta_logits.nii.gz"
        mask_path = mask_root / f"{case}/ses-02/{case}_ses-02_space-ncct_lesion-msk.nii.gz"
        mask = nib.load(mask_path)

        logits = []
        for dir in sweep_dirs:
            logits.append(nib.load(dir / logit_name).get_fdata())
        logits_mean = np.mean(np.stack(logits), axis=0)

        logits_dst_path = dst_dir / logit_name
        logits_dst_path.parent.mkdir(exist_ok=True, parents=True)
        logits_img = nib.Nifti1Image(logits_mean, affine=mask.affine, header=mask.header)
        nib.save(logits_img, logits_dst_path)

        pred = softmax(logits_mean, axis=-1)[..., 1] > 0.5
        pred_path = dst_dir / logit_name.replace("logits", "pred")
        pred_path.parent.mkdir(exist_ok=True, parents=True)
        pred_img = nib.Nifti1Image(pred, affine=mask.affine, header=mask.header)
        nib.save(pred_img, pred_path)

        # Voxel size in mL (convert from mm^3)
        voxel_spacing = np.array(mask.header.get_zooms())
        voxel_size = np.prod(voxel_spacing) / 1000

        label = mask.get_fdata().astype(int)

        # Compute metrics
        abs_vol_diff = compute_absolute_volume_difference(
            label, pred, voxel_size
        )
        f1_score, instance_count_diff, dice_score = (
            compute_dice_f1_instance_difference(label, pred)
        )

        results.append(
            {
                "case_id": case,
                "dice": dice_score,
                "f1_score": f1_score,
                "abs_vol_diff": abs_vol_diff,
                "instance_count_diff": instance_count_diff,
            }
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(dst_dir / "results.csv", index=False)

if __name__ == "__main__":
    main()
