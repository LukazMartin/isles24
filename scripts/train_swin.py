"""
Train multi encoder Swin-UNETR
"""

from pathlib import Path
import re
import argparse
from dataclasses import asdict
import wandb
from isles.swin.config import SwinTrainConfig
from isles.swin.model import get_model
from isles.swin.training import train_swin, get_swin_dataloaders
from isles.swin.evaluation import final_evaluation
from isles.utils import generate_datalist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multi encoder Swin-UNETR")
    parser.add_argument("--run-id", required=True, type=str)
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=["cta", "cbf"],
        type=str,
        choices=["cta", "cbf", "cbv", "mtt", "tmax"],
    )
    parser.add_argument(
        "--model",
        default="MultiEncoderSwinUNETR",
        type=str,
        choices=["BaseSwinUNETR", "MultiEncoderSwinUNETR"],
    )
    parser.add_argument(
        "--crop-mode",
        default="label_classes",
        type=str,
        choices=["label_classes", "spatial"],
    )
    parser.add_argument("--max-epochs", default=300, type=int)
    parser.add_argument("--learning-rate", default=1e-4, type=float)
    parser.add_argument("--evaluation", action="store_true")
    parser.add_argument(
        "--include-tabular-embeddings",
        action="store_true",
        help="Use precomputed TabPFN tabular embeddings as an extra conditioning input.",
    )
    parser.add_argument(
        "--tabular-embeddings-dir",
        type=Path,
        default=Path("artifacts/tabpfn_embeddings"),
        help=(
            "Directory containing phenotype_embeddings.npy and "
            "phenotype_embedding_index.csv."
        ),
    )
    return parser.parse_args()

def _case_id_from_path(path: str) -> str:
    match = re.search(r"sub-stroke\d+", path)
    if match is None:
        raise ValueError(f"Could not parse case id from path: {path}")
    return match.group()


def attach_tabular_embeddings(datalist: dict, embedding_dir: Path) -> int:
    import numpy as np
    import pandas as pd

    embeddings = np.load(embedding_dir / "phenotype_embeddings.npy")
    index_df = pd.read_csv(embedding_dir / "phenotype_embedding_index.csv")

    if len(index_df) != embeddings.shape[0]:
        raise ValueError(
            "Embedding index size does not match embedding rows: "
            f"{len(index_df)} vs {embeddings.shape[0]}"
        )

    vectors_by_case: dict[str, list[np.ndarray]] = {}
    for row_idx, row in index_df.iterrows():
        case = row["subject_id"]
        if not isinstance(case, str):
            continue
        vectors_by_case.setdefault(case, []).append(embeddings[row_idx])

    vector_by_case = {
        case: np.mean(np.stack(vectors, axis=0), axis=0).astype(np.float32)
        for case, vectors in vectors_by_case.items()
    }

    assigned = 0
    for split in ("training", "validation", "testing"):
        for sample in datalist.get(split, []):
            case = _case_id_from_path(sample["label"])
            if case not in vector_by_case:
                raise KeyError(f"No tabular embedding found for case {case}")
            sample["tabular_embedding"] = vector_by_case[case]
            assigned += 1

    embedding_dim = next(iter(vector_by_case.values())).shape[0]
    print(
        f"Attached tabular embeddings to {assigned} samples; embedding_dim={embedding_dim}"
    )
    return embedding_dim

def main() -> None:
    args = parse_args()

    config = SwinTrainConfig(
        model=args.model,
        max_epochs=args.max_epochs,
        modalities=args.modalities,
        target_spacing=(1.0, 1.0, 1.0),
        roi_size=(64, 64, 64),
        learning_rate=args.learning_rate,
        crop_mode=args.crop_mode,
        crop_ratios=(1, 1),
        include_background=False,
        intensity_windows={
            "cta": [0, 90],
            "cbf": [0, 35],
            "cbv": [0, 10],
            "mtt": [0, 20],
            "tmax": [0, 7],
        },
        batch_size=1,
        val_interval=10,
        inspect_training=True,
        inspect_interval=25,
    )

    data_root = Path("/home/renku/work/data-local")
    pretrained_path = (
        data_root / "pretrained/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt"
    )
    run_dir = data_root / f"runs/{args.run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    datalist = generate_datalist(
        data_root=data_root,
        target_dir=run_dir,
        modalities=config.modalities,
        brain_mask=True,
        val_fold=0,
    )
    if args.include_tabular_embeddings:
        embedding_dir = args.tabular_embeddings_dir
        if not embedding_dir.is_absolute():
            embedding_dir = Path.cwd() / embedding_dir
        config.tabular_embedding_dim = attach_tabular_embeddings(
            datalist=datalist, embedding_dir=embedding_dir
        )

    if args.evaluation and args.include_tabular_embeddings:
        raise NotImplementedError(
            "Final evaluation with tabular conditioning is not implemented yet. "
            "Train with --include-tabular-embeddings and evaluate separately later."
        )
    wandb.init(
        project="ISLES",
        name=args.run_id,
        dir=run_dir,
        config={
            **asdict(config),
            "loss": "DiceCELoss",
            "optimizer": "AdamW",
            "scheduler": "WarmupCosineSchedule",
        },
        save_code=True,
    )
    artifact = wandb.Artifact("datalist", type="datalist")
    artifact.add_file(run_dir / "datalist.json", name="datalist.json")
    wandb.log_artifact(artifact)

    train_loader, val_loader = get_swin_dataloaders(datalist, config)
    model = get_model(config)
    model.load_pretrained_encoders(pretrained_path)

    train_swin(
        model=model,
        config=config,
        run_dir=run_dir,
        train_loader=train_loader,
        val_loader=val_loader,
        case_id_fn=lambda p: re.search(r"sub-stroke\d+", p).group(),
    )

    if args.evaluation:
        checkpoint_path = run_dir / "checkpoints/best_model.pt"
        eval_dir = run_dir / "evaluation"
        final_evaluation(
            checkpoint_path=checkpoint_path,
            val_loader=val_loader,
            config=config,
            out_dir=eval_dir,
            save_logits=True,
        )

        wandb.save(f"{eval_dir}/**/*", base_path=run_dir)
    wandb.finish()


if __name__ == "__main__":
    main()
