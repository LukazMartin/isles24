#!/usr/bin/env python3
"""Create TabPFN embeddings for ISLES phenotype Excel files.

Expected layout (typical):
- data-local/train/phenotype/sub-strokeXXXX/ses-01/*_demographic_baseline.xlsx
- train/phenotype/sub-strokeXXXX/ses-01/*_demographic_baseline.xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract TabPFN embeddings from phenotype baseline Excel files."
    )
    parser.add_argument(
        "--phenotype-root",
        type=Path,
        default=Path("train/phenotype"),
        help=(
            "Root directory containing phenotype files. "
            "If not found, the script also tries common alternatives like "
            "data-local/train/phenotype."
        ),
    )
    parser.add_argument(
        "--session",
        type=str,
        default="ses-01",
        help="Session folder to read (default: ses-01).",
    )
    parser.add_argument(
        "--glob",
        dest="file_glob",
        type=str,
        default="*_demographic_baseline.xlsx",
        help="Excel file pattern inside each session folder.",
    )
    parser.add_argument(
        "--n-fold",
        type=int,
        default=0,
        help="Number of CV folds for TabPFNEmbedding (0 = vanilla embeddings).",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=1,
        help="Number of estimators for TabPFNRegressor.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state passed to TabPFNRegressor.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/tabpfn_embeddings"),
        help="Directory where embeddings and metadata are written.",
    )
    return parser.parse_args()


def resolve_candidate_roots(phenotype_root: Path) -> list[Path]:
    """Find likely phenotype roots based on current project layouts."""

    cwd = Path.cwd()
    candidates = [
        phenotype_root,
        cwd / phenotype_root,
        cwd / "data-local/train/phenotype",
        cwd.parent / "data-local/train/phenotype",
        cwd / "../data-local/train/phenotype",
    ]

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            deduped.append(resolved)

    return deduped


def find_excel_paths(phenotype_root: Path, session: str, file_glob: str) -> tuple[list[Path], Path | None]:
    roots = resolve_candidate_roots(phenotype_root)
    patterns = [
        f"sub-stroke*/{session}/{file_glob}",
        f"*/{session}/{file_glob}",
        f"**/{session}/{file_glob}",
    ]

    for root in roots:
        if not root.exists():
            continue

        matches: list[Path] = []
        for pattern in patterns:
            matches.extend(root.glob(pattern))

        unique_matches = sorted({p.resolve() for p in matches})
        if unique_matches:
            return unique_matches, root

    return [], None


def load_phenotype_table(phenotype_root: Path, session: str, file_glob: str) -> tuple[Any, Path]:
    import pandas as pd

    excel_paths, matched_root = find_excel_paths(phenotype_root, session, file_glob)
    if not excel_paths or matched_root is None:
        searched = "\n".join([f"- {p}" for p in resolve_candidate_roots(phenotype_root)])
        raise FileNotFoundError(
            "No files were found for the requested phenotype layout.\n"
            f"Expected pattern: */{session}/{file_glob}\n"
            "Searched roots:\n"
            f"{searched}\n"
            "If your data is under data-local/train/phenotype, run:\n"
            "  python scripts/tabpfn_embed_phenotype.py --phenotype-root ../data-local/train/phenotype"
        )

    tables: list[Any] = []
    for path in excel_paths:
        frame = pd.read_excel(path)
        frame = frame.copy()
        frame["subject_id"] = path.parent.parent.name
        frame["session"] = path.parent.name
        frame["source_file"] = str(path)
        tables.append(frame)

    table = pd.concat(tables, ignore_index=True)

    for col in table.columns:
        if pd.api.types.is_datetime64_any_dtype(table[col]):
            table[col] = table[col].astype("string")

    return table, matched_root


def build_features(table: Any) -> Any:
    metadata_cols = {"subject_id", "session", "source_file"}
    X = table.drop(columns=[col for col in metadata_cols if col in table.columns])

    if X.shape[1] == 0:
        raise ValueError("No feature columns found after dropping metadata columns.")

    return X


def build_pseudo_targets(num_rows: int) -> Any:
    """Create deterministic pseudo-targets required by the TabPFN embedding API."""

    import numpy as np

    return np.arange(num_rows, dtype=np.float32)


def normalize_embeddings(embeddings: Any) -> Any:
    """Convert TabPFN embedding output to [n_samples, emb_dim]."""

    import numpy as np

    arr = np.asarray(embeddings)

    if arr.ndim == 2:
        return arr

    # Common case from TabPFNEmbedding: [n_estimators, n_samples, emb_dim].
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            return arr[0]
        return arr.mean(axis=0)

    raise ValueError(
        f"Unexpected embedding shape {arr.shape}. Expected 2D or 3D tensor."
    )


def main() -> None:
    args = parse_args()

    try:
        import numpy as np
        import pandas as pd
        from tabpfn_extensions import TabPFNRegressor
        from tabpfn_extensions.embedding import TabPFNEmbedding
    except ImportError as exc:
        raise ImportError(
            "Missing dependency. Install with: pip install 'tabpfn-extensions[embedding]'"
        ) from exc

    phenotype, matched_root = load_phenotype_table(args.phenotype_root, args.session, args.file_glob)
    X = build_features(phenotype)
    y_pseudo = build_pseudo_targets(len(phenotype))

    model = TabPFNRegressor(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
    )
    extractor = TabPFNEmbedding(tabpfn_clf=model, n_fold=args.n_fold)

    extractor.fit(X, y_pseudo)
    embeddings_raw = extractor.get_embeddings(X, y_pseudo, X, data_source="test")
    embeddings = normalize_embeddings(embeddings_raw)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "phenotype_embeddings.npy", embeddings)
    np.save(args.output_dir / "phenotype_embeddings_raw.npy", embeddings_raw)

    metadata = phenotype[["subject_id", "session", "source_file"]].copy()
    metadata["embedding_row"] = np.arange(len(metadata))
    metadata.to_csv(args.output_dir / "phenotype_embedding_index.csv", index=False)

    pd.DataFrame(embeddings).to_csv(
        args.output_dir / "phenotype_embeddings.csv", index=False
    )

    print(f"Using phenotype root: {matched_root}")
    print("No ground-truth tabular target found; using deterministic pseudo-targets (row indices).")
    print(f"Loaded rows: {len(phenotype)}")
    print(f"Feature columns: {X.shape[1]}")
    print(f"Raw embedding shape: {embeddings_raw.shape}")
    print(f"Normalized embedding shape: {embeddings.shape}")
    print(f"Saved files under: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
