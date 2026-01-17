from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

from milasol.data.datasets_new import ProteinDataset, collate_fn
from milasol.models.base import ProteinClassifier
from milasol.data.seq_to_embeddings_new import transfer_to_three_embedding_csvs
from milasol.data.seq_to_embeddings_new import EmbeddingArtifacts


@torch.no_grad()
def get_pred(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []

    t0 = time.time()
    n_batches = len(loader) if hasattr(loader, "__len__") else None
    n_seen = 0
    log_every = 50
    for step, (seqs, esm_embs, prot_embs, raygun_embs, feats, labels, _) in enumerate(
        loader, start=1
    ):
        # move to device
        seqs = seqs.to(device)
        esm_embs = esm_embs.to(device)
        prot_embs = prot_embs.to(device)
        raygun_embs = raygun_embs.to(device)

        # forward
        logits, _, _, _ = model(seqs, esm_embs, prot_embs, raygun_embs)
        probs = torch.sigmoid(logits).view(-1)
        preds = (probs > 0.5).to(torch.int64).view(-1)

        bs = int(probs.numel())
        n_seen += bs

        all_probs.extend(probs.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

        if labels is not None:
            flat_labels = torch.as_tensor(labels).detach().cpu().view(-1).tolist()
            all_labels.extend(flat_labels)
        else:
            all_labels.extend([None] * probs.numel())

        # processing record
        if log_every > 0 and (step % log_every == 0 or step == 1):
            dt = time.time() - t0
            rate = n_seen / max(dt, 1e-9)
            batch_info = f"{step}/{n_batches}" if n_batches is not None else str(step)
            print(
                f"[get_pred] {batch_info} batches | seen={n_seen} | rate={rate:.1f} seqs/s",
                flush=True,
            )

    print(
        f"[get_pred] done | total_seen={n_seen} | elapsed={time.time()-t0:.2f}s",
        flush=True,
    )
    return all_probs, all_preds, all_labels


def _normalize_device(dev: Union[str, torch.device]) -> torch.device:
    if isinstance(dev, str):
        if dev.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(dev)
    return dev


def init_model(modelname: str, device: Union[str, torch.device]):
    device = _normalize_device(device)
    model = ProteinClassifier(
        vocab_size=22,
        embed_dim=128,
        num_filters=128,
        kernel_size=6,
        lstm_hidden_dim=128,
        num_lstm_layers=1,
        esm_dim=1280,
        prot_dim=1024,
        output_dim=1,
        latent_dim=64,
    ).to(device)

    state_dict = torch.load(modelname, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _build_dataset_from_artifacts(
    artifacts: EmbeddingArtifacts,
    label_source: Optional[Union[str, Sequence, torch.Tensor]] = None,
) -> ProteinDataset:
    sequence_source = artifacts.sequence_path or artifacts.sequences
    esm_source = artifacts.esm_path or artifacts.esm_embeddings
    prot_source = artifacts.prott5_path or artifacts.prott5_embeddings
    ray_source = artifacts.raygun_path or artifacts.raygun_embeddings

    return ProteinDataset(
        sequence_source,
        esm_source,
        prot_source,
        ray_source,
        feats_file=None,
        label_file=label_source,
        augment=False,
    )


def prediction(
    model_or_path: Union[torch.nn.Module, str, Path],
    source_data: Optional[Union[str, Sequence[str]]] = None,
    *,
    device: Union[str, torch.device] = "cpu",
    out_dir: Union[str, Path, None] = "outputs/",
    cache_dir: Union[str, None] = None,
    batch_size: int = 32,
    write_to_disk: bool = False,
    labels: Optional[Union[str, Sequence, torch.Tensor]] = None,
    precomputed_inputs: Optional[Dict[str, Union[str, Path, torch.Tensor]]] = None,
) -> Tuple[Sequence[float], Sequence[int], Sequence[Optional[int]]]:
    device = _normalize_device(device)

    if isinstance(model_or_path, (str, Path)):
        model = init_model(str(model_or_path), device=device)
    else:
        model = model_or_path.to(device)

    if precomputed_inputs is not None:
        if write_to_disk:
            raise ValueError(
                "write_to_disk is only supported when generating embeddings from raw sequences."
            )

        seq_input = precomputed_inputs.get("sequence") or precomputed_inputs.get(
            "sequence_file"
        )
        esm_input = precomputed_inputs.get("esm") or precomputed_inputs.get("esm_file")
        prot_input = precomputed_inputs.get("prot") or precomputed_inputs.get(
            "prot_file"
        )
        ray_input = precomputed_inputs.get("ray") or precomputed_inputs.get("ray_file")

        missing = [
            name
            for name, value in (
                ("sequence", seq_input),
                ("esm", esm_input),
                ("prot", prot_input),
                ("ray", ray_input),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "Missing required precomputed inputs: " + ", ".join(missing)
            )

        dataset = ProteinDataset(
            seq_input,
            esm_input,
            prot_input,
            ray_input,
            feats_file=None,
            label_file=labels,
            augment=False,
        )
    else:
        if source_data is None:
            raise ValueError(
                "source_data must be provided when precomputed_inputs is not supplied."
            )

        out_path = Path(out_dir) if (write_to_disk and out_dir is not None) else None
        if out_path is not None:
            out_path.mkdir(parents=True, exist_ok=True)

        artifacts = transfer_to_three_embedding_csvs(
            source_data,
            seq_col="sequence",
            out_dir=str(out_path) if out_path is not None else None,
            cache_dir=cache_dir,
            write_to_disk=write_to_disk,
        )

        dataset = _build_dataset_from_artifacts(artifacts, label_source=labels)

    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    all_probs, all_preds, all_labels = get_pred(model, loader, device)

    cleaned_labels: list[Optional[int]] = []
    for val in all_labels:
        if val is None:
            cleaned_labels.append(None)
            continue
        try:
            v = int(round(float(val)))
        except (TypeError, ValueError):
            cleaned_labels.append(None)
            continue
        cleaned_labels.append(v if v >= 0 else None)

    return all_probs, all_preds, cleaned_labels


def main():
    ap = argparse.ArgumentParser(description="Run protein solubility prediction")
    ap.add_argument("--modelname", required=True, help="Checkpoint file")
    ap.add_argument(
        "--source_data",
        nargs="+",
        default=None,
        help="Input: CSV/TXT path, single sequence, or multiple sequences",
    )
    ap.add_argument("--out_dir", default="outputs/")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument(
        "--write_to_disk",
        action="store_true",
        help="Persist intermediate embeddings to CSV/text files",
    )
    ap.add_argument(
        "--label_file",
        default=None,
        help="Optional label file or tensor source for evaluation mode",
    )
    ap.add_argument(
        "--sequence_file", default=None, help="Precomputed sequence file (txt/csv)"
    )
    ap.add_argument("--esm_file", default=None, help="Precomputed ESM embedding file")
    ap.add_argument(
        "--prot_file", default=None, help="Precomputed ProtT5 embedding file"
    )
    ap.add_argument(
        "--ray_file", default=None, help="Precomputed RayGun embedding file"
    )
    args = ap.parse_args()

    source = None
    if args.source_data:
        source = args.source_data[0] if len(args.source_data) == 1 else args.source_data

    precomputed_inputs = None
    precomputed_fields = [
        args.sequence_file,
        args.esm_file,
        args.prot_file,
        args.ray_file,
    ]
    if any(precomputed_fields):
        if not all(precomputed_fields):
            ap.error(
                "Must provide --sequence_file, --esm_file, --prot_file, and --ray_file together when using precomputed embeddings."
            )
        precomputed_inputs = {
            "sequence": args.sequence_file,
            "esm": args.esm_file,
            "prot": args.prot_file,
            "ray": args.ray_file,
        }

    if source is None and precomputed_inputs is None:
        ap.error(
            "Either --source_data or all precomputed embedding files must be provided."
        )

    labels_arg = args.label_file
    print("Loading model...")
    model = init_model(args.modelname, device=args.device)
    probs, preds, labels_out = prediction(
        model,
        source,
        device=args.device,
        out_dir=args.out_dir,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        write_to_disk=args.write_to_disk,
        labels=labels_arg,
        precomputed_inputs=precomputed_inputs,
    )
    print("Saving predictions...")
    output_dir = Path(args.out_dir) if args.out_dir else Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)
    df_dict = {
        "predicted_prob": probs,
        "predicted_label": preds,
    }
    if any(lbl is not None for lbl in labels_out):
        df_dict["true_label"] = labels_out
    pd.DataFrame(df_dict).to_csv(output_dir / "predictions.csv", index=False)
    print("Prediction is completed.")


if __name__ == "__main__":
    main()
