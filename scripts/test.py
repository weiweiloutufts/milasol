import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


# ============= SET SEEDS FIRST =============
def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)


set_all_seeds(42)
# ============= END SEED SETTING =============

cache_dir = "/cluster/tufts/cowenlab/wlou01/modelcache"
os.environ["TORCH_HOME"] = cache_dir
os.environ.setdefault("HF_HOME", cache_dir)
torch.hub.set_dir(cache_dir)

from milasol.models.predict_new import prediction, init_model


# --------- helpers ----------
def to_np(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)


def print_metrics(all_probs, all_preds, all_labels):
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(
        all_labels, all_preds, average="binary", zero_division=0
    )
    recall = recall_score(all_labels, all_preds, average="binary", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)
    mcc = matthews_corrcoef(all_labels, all_preds)

    # Probabilities needed for AUROC and AUPRC
    if all_probs is not None:
        auroc = roc_auc_score(all_labels, all_probs)
        auprc = average_precision_score(all_labels, all_probs)
    else:
        auroc = auprc = 0.0  # or handle None as needed

    # Confusion matrix to get TN, FP, FN, TP
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

    # Sensitivity (Insoluble) = TN / (TN + FP)
    sensitivity_insoluble = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Precision (Insoluble) = TN / (TN + FN)
    precision_insoluble = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    # Gain (Soluble) = Recall / (Proportion of soluble in dataset)
    total = tp + tn + fp + fn
    n_soluble = tp + fn
    n_insoluble = tn + fp
    gain_soluble = (recall / (n_soluble / total)) if n_soluble > 0 else 0.0
    gain_insoluble = (
        (sensitivity_insoluble / (n_insoluble / total)) if n_insoluble > 0 else 0.0
    )

    # Print all metrics
    print(
        f"ACC: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, "
        f"F1: {f1:.4f}, MCC: {mcc:.4f}, AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}, "
        f"Sens(Insoluble): {sensitivity_insoluble:.4f}, Prec(Insoluble): {precision_insoluble:.4f}, "
        f"Gain(Sol): {gain_soluble:.2f}, Gain(Ins): {gain_insoluble:.2f}"
    )


# ----------------------------

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "../checkpoints/daps_train0.3_0.1_0.1_0.0_64_2.5_1.0.pth"
fit_model = init_model(modelname=model_path, device=device)
fit_model.eval()

# Load data
data_path = "test_src.txt"
scr_data = pd.read_csv(
    data_path, header=None, names=["sequence"], dtype=str, keep_default_na=False
)
seqs = scr_data["sequence"].astype(str).str.strip()
seqs = seqs[seqs.str.len() > 0].tolist()
print(f"Total sequences: {len(seqs)}\n")
tgt_file = "test_tgt.txt"
tgt_data = pd.read_csv(
    tgt_file, header=None, names=["labels"], dtype=str, keep_default_na=False
)
labels = tgt_data["labels"].astype(int).tolist()


print("\n" + "=" * 50)
print("In chunks of 32")
print("=" * 50)

chunk_size = 32
all_probs, all_preds = [], []

for i in range(0, len(seqs), chunk_size):
    chunk = seqs[i : i + chunk_size]
    probs_chunk, preds_chunk, _ = prediction(
        fit_model,
        chunk,
        device=device,
        out_dir="test_outputs",
        cache_dir=cache_dir,
        write_to_disk=False,
    )
    all_probs.extend(to_np(probs_chunk))
    all_preds.extend(to_np(preds_chunk))

print("\nMetrics:")
print_metrics(all_probs, all_preds, labels)
