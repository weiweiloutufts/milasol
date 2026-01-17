import pandas as pd
import torch
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
from typing import Union
from milasol.data.datasets import ProteinDataset, collate_fn
from milasol.models.base import ProteinClassifier


@torch.no_grad()
def get_pred(model, loader, device):
    model.eval()
    all_probs, all_preds, all_index, all_labels = [], [], [], []

    for seqs, esm_embs, prot_embs, raygun_embs, feats, labels, idx in loader:
        # feats/labels may be sentinel tensors; be defensive
        seqs = seqs.to(device)
        esm_embs = esm_embs.to(device)
        prot_embs = prot_embs.to(device)
        raygun_embs = raygun_embs.to(device)

        logits, _, _, _ = model(seqs, esm_embs, prot_embs, raygun_embs)  # (B,) or (B,1)

        probs = torch.sigmoid(logits).view(-1)  # (B,)
        preds = (probs > 0.5).to(torch.int64).view(-1)

        all_probs.extend(probs.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

        if labels is not None:
            all_labels.extend(torch.as_tensor(labels).flatten().cpu().tolist())

    return all_probs, all_preds, all_labels


def _normalize_device(dev: Union[str, torch.device, None]) -> torch.device:
    if dev is None or (isinstance(dev, str) and dev.strip() == ""):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(dev, str):
        if dev.startswith("cuda") and not torch.cuda.is_available():
            # fall back gracefully if someone passes "cuda" but no CUDA present
            return torch.device("cpu")
        return torch.device(dev)

    if isinstance(dev, torch.device):
        if dev.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return dev

    raise TypeError("device must be str, torch.device, or None")


def init_model(modelname: str, device: torch.device | str):

    #  instantiate the architecture (keep these hyperparams in sync with training)
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

    #  load weights
    state_dict = torch.load(modelname, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    # print(f"[INFO] Loaded model weights from {ckpt_path}")
    return model


def prediction(
    model_name: str,
    source_dir: str,
    device: str | torch.device | None = None,
    out_dir: str = "outputs/",
    cache_dir: str | None = None,
):
    device = _normalize_device(device)
    print(f"[INFO] Using device: {device}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source_data_txt = Path(source_dir) / "test_src.txt"
    esm_csv = Path(source_dir) / "test_src_esm_embeddings.csv"
    prott5_csv = Path(source_dir) / "test_src_prot_embeddings.csv"
    ray_csv = Path(source_dir) / "test_src_raygun_embeddings.csv"
    label_file = Path(source_dir) / "test_tgt.txt"

    test_data = ProteinDataset(
        source_data_txt,
        esm_csv,
        prott5_csv,
        ray_csv,
        feats_file=None,
        label_file=label_file,
        augment=False,
    )

    test_loader = DataLoader(
        test_data, batch_size=32, shuffle=False, drop_last=False, collate_fn=collate_fn
    )

    model = init_model(model_name, device)

    # predict
    all_probs, all_preds, all_labels = get_pred(model, test_loader, device)

    # save alongside the source CSV
    # read one sequence per line, keep "NA" as literal, strip blanks
    seq_df = pd.read_csv(
        source_data_txt, header=None, names=["seq"], dtype=str, keep_default_na=False
    )
    seq_df["seq"] = seq_df["seq"].str.strip()

    # sanity checks

    out_csv = str(Path(out_dir) / "test_predictions.csv")
    pd.DataFrame(
        {
            "seq": seq_df["seq"],
            "predicted_label": all_preds,
            "predicted_prob": all_probs,
            "true_label": all_labels,
        }
    ).to_csv(out_csv, index=False)

    print(f"[INFO] Predictions saved to {out_csv}")


def main():
    ap = argparse.ArgumentParser(description="Run protein solubility prediction")
    ap.add_argument("--modelname", required=True, help="Checkpoint file")
    ap.add_argument("--source_dir", required=True, help="source file directory")

    ap.add_argument("--out_dir", default="outputs/")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument(
        "--device", default=None, help="Device for inference, e.g., 'cuda:0' or 'cpu'"
    )
    args = ap.parse_args()

    print("[INFO] predict_pre starting...", flush=True)
    print(f"[INFO] args: {args}", flush=True)

    prediction(
        args.modelname,
        source_dir=args.source_dir,
        device=args.device,
        out_dir=args.out_dir,
        cache_dir=args.cache_dir,
    )

    print("[INFO] predict_pre done.", flush=True)


if __name__ == "__main__":
    main()
