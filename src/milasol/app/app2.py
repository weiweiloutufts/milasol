import os, time, math, csv, numpy as np, torch
from typing import List, Optional


cache_dir = "/cluster/tufts/cowenlab/wlou01/modelcache"
os.environ["TORCH_HOME"] = cache_dir
os.environ.setdefault("HF_HOME", cache_dir)
torch.hub.set_dir(cache_dir)

from milasol.models.predict_new import prediction, init_model

# from raygun.pretrained import raygun_4_4mil_800M
from esm.pretrained import esm2_t33_650M_UR50D
from tqdm import tqdm
from torch.utils.data import DataLoader
from raygun.modelv2.loader import RaygunData
import pandas as pd
import random
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from pathlib import Path
from Bio import SeqIO
from io import StringIO
from datetime import datetime


def prep_data(
    seq_list: List[str],
    prefix: str = "seq",
    start_index: int = 1,
    uppercase: bool = True,
) -> List[SeqRecord]:

    recs: List[SeqRecord] = []
    for idx, s in enumerate(seq_list, start_index):
        if s is None:
            continue
        s = str(s).strip()
        if not s:
            continue
        if uppercase:
            s = s.upper()
        recs.append(
            SeqRecord(Seq(s), id=f"{prefix}{idx}", description=f"{prefix}_{idx}")
        )

    # print(f"Total SeqRecords: {len(recs)}")

    return recs


def seqs_to_fasta_handle(
    seq_list: List[str],
    **prep_kwargs,
) -> StringIO:
    """
    Build an in-memory FASTA file-like handle from sequences.
    Use this with RaygunData, which expects a file path/handle.
    """
    recs = prep_data(seq_list, **prep_kwargs)
    handle = StringIO()
    SeqIO.write(recs, handle, "fasta")
    handle.seek(0)
    return handle


def _latent_distance(
    initial_latent: torch.Tensor, proposal_latent: torch.Tensor
) -> float:
    """
    Compute L2 distance between two latent embeddings, padding the shorter sequence length.
    """
    if initial_latent.shape == proposal_latent.shape:
        diff = proposal_latent - initial_latent
        return float(torch.norm(diff.flatten(), p=2).item())

    dim = initial_latent.shape[1]
    len_init = initial_latent.shape[0]
    len_prop = proposal_latent.shape[0]
    target_len = max(len_init, len_prop)

    def _pad_latent(latent: torch.Tensor, target: int) -> torch.Tensor:
        if latent.shape[0] == target:
            return latent
        pad_rows = target - latent.shape[0]
        pad_tensor = torch.zeros((pad_rows, latent.shape[1]), device=latent.device)
        return torch.cat([latent, pad_tensor], dim=0)

    padded_init = _pad_latent(initial_latent, target_len)
    padded_prop = _pad_latent(proposal_latent, target_len)
    diff = padded_prop - padded_init
    return float(torch.norm(diff.flatten(), p=2).item())


def generate_with_raygun(
    seq_list,
    raymodel,
    esm_model,
    alphabet,
    proposal_std,
    device=None,
    batch_size=32,
    shuffle=False,
    return_names=False,
):

    raymodel.eval()
    if device is None:
        try:
            device = next(raymodel.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raymodel.to(device)
    esm_model.eval().to(device)

    # print(f"[Generate with Raygun] length of seq_list is : {len(seq_list)}")
    # --- build loader WITHOUT recreating ESM each call ---
    fasta_handle = seqs_to_fasta_handle(seq_list, prefix="b0_seq", start_index=1)
    preddata = RaygunData(
        fasta_handle, alphabet, esm_model, device=device, maxlength=2000
    )
    fasta_handle.close()

    # print(f"[Generate with Raygun] Total sequences for generation: {len(preddata)}")
    predloader = DataLoader(
        preddata,
        shuffle=shuffle,
        batch_size=batch_size,
        collate_fn=preddata.collatefn,
        pin_memory=(device.type == "cuda"),
        num_workers=0,
    )

    # --- inference context (no grads + safe autocast on CUDA) ---
    use_autocast = device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if (use_autocast and torch.cuda.is_bf16_supported())
        else torch.float16
    )

    pred_seqs, names = [], []
    latents = []
    with torch.inference_mode():
        ctx = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if use_autocast
            else torch.amp.autocast("cpu", enabled=False)
        )
        with ctx:
            for tok, emb, mask, bat in tqdm(predloader, desc="Raygun generation"):
                tok = tok.to(device, non_blocking=True)
                emb = emb.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)

                names_batch = []
                if bat and isinstance(bat[0], (list, tuple)) and len(bat[0]) >= 1:
                    bn, _true = zip(*bat)
                    names_batch = list(bn)
                    names.extend(names_batch)

                noise = None
                if proposal_std and proposal_std > 0:
                    noise = float(proposal_std)

                # print(f"[Generate with Raygun] Running generation on batch size: {emb.size(0)}")

                out = raymodel(
                    emb,
                    mask=mask,
                    noise=noise,
                    return_logits_and_seqs=True,
                )

                # print(f"[Generate with Raygun] Length of out is {len(out)}")
                # normalize generated sequences per sample
                gen_key = "generated-sequences"
                gen = out[gen_key]

                batch_size = emb.size(0)

                # capture latent embeddings with validation
                flemb = out["fixed_length_embedding"]

                # print(f"[Generate with Raygun] Captured fixed_length_embedding with shape {flemb.shape} for batch size {batch_size}")

                if not torch.is_tensor(flemb):
                    raise RuntimeError(
                        "Raygun output missing `fixed_length_embedding` tensor."
                    )
                if (
                    flemb.ndim != 3
                    or flemb.shape[0] != batch_size
                    or flemb.shape[1] != 50
                    or flemb.shape[2] != 1280
                ):
                    raise RuntimeError(
                        f"Raygun `fixed_length_embedding` expected shape "
                        f"({batch_size}, 50, 1280); got {tuple(flemb.shape)}"
                    )

                gen_count = len(gen) if isinstance(gen, (list, tuple)) else 1
                flemb_count = flemb.shape[0]

                actual_count = min(batch_size, gen_count, flemb_count)
                if actual_count < batch_size:
                    dropped_names = (
                        names_batch[actual_count:batch_size] if names_batch else []
                    )
                    print(
                        f"[Raygun] Warning: generator returned fewer outputs than inputs "
                        f"(in={batch_size}, gen={gen_count}, emb_out={flemb_count}); "
                        f"dropped inputs: {dropped_names if dropped_names else 'unknown inputs'}"
                    )
                    raise RuntimeError("Raygun dropped inputs; aborting run.")

                if isinstance(gen, (list, tuple)):
                    pred_seqs.extend([str(s) for s in gen[:actual_count]])
                elif torch.is_tensor(gen):
                    raise TypeError(
                        "Raygun returned token IDs (Tensor). Provide a decoder to convert to strings."
                    )
                else:
                    pred_seqs.append(str(gen))

                for i in range(actual_count):
                    latents.append(flemb[i].detach().cpu())

    expected_total = len(seq_list)
    if len(latents) != expected_total:
        expected_names = [f"b0_seq{i}" for i in range(1, expected_total + 1)]
        missing = [
            (name, seq_list[idx])
            for idx, name in enumerate(expected_names)
            if name not in names
        ]
        print(
            f"[Raygun] Error: received {len(latents)} outputs for {expected_total} inputs. "
            f"Missing inputs: {missing if missing else 'unknown'}"
        )
        raise RuntimeError("Raygun dropped inputs; aborting run.")

    if return_names:
        return pred_seqs, latents, names
    return pred_seqs, latents


def protein_design_batch(
    fit_model: torch.nn.Module,
    initial_sequences: List[str],
    *,
    raymodel: torch.nn.Module,
    esm_model: torch.nn.Module,
    alphabet,
    n_steps: int = 20,
    initial_temp: float = 1.0,
    final_temp: float = 1e-3,
    proposal_std: float = 0.1,
    device: Optional[torch.device] = None,
    out_dir: str = "outputs",
    cache_dir: str = "/cluster/tufts/cowenlab/wlou01/modelcache/",
    log_every: int = 50,
    step_log_csv: Optional[str] = None,  # e.g., "outputs/sa_steps.csv"
    gen_batch_size: Optional[int] = None,  # cap Raygun batch size if needed
    clip_exp: float = 100.0,  # clamp for exp() to avoid overflow
    n_restarts: int = 100,  # number of times to restart the annealing loop
    latent_distance_cap: float = 15,  # sigma x root(dim) * 4 = 0.1 x sqrt(1280) * 4
):

    # ---- local helper: normalize prediction() outputs to 1D float list ----
    def _to_1d_float_list(x) -> List[float]:
        import numpy as _np

        if torch.is_tensor(x):
            return [float(v) for v in x.detach().reshape(-1).tolist()]
        if isinstance(x, _np.ndarray):
            return [float(v) for v in x.reshape(-1).tolist()]
        if isinstance(x, (list, tuple)):
            out = []
            for v in x:
                if torch.is_tensor(v):
                    out.extend([float(u) for u in v.detach().reshape(-1).tolist()])
                elif isinstance(v, _np.ndarray):
                    out.extend([float(u) for u in v.reshape(-1).tolist()])
                else:
                    out.append(float(v))
            return out
        return [float(x)]

    assert len(initial_sequences) > 0, "Need at least one sequence."
    B = len(initial_sequences)

    # ---- device/model prep ----
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    raymodel.eval().to(device)
    esm_model.eval().to(device)

    # ---- score initial batch ----
    curr_seqs = list(initial_sequences)
    init_probs, _, _ = prediction(
        fit_model,
        curr_seqs,
        device=device,
        out_dir=out_dir,
        cache_dir=cache_dir,
        write_to_disk=False,
    )
    curr_fit = np.array(_to_1d_float_list(init_probs), dtype=float)  # [B]
    best_fit = curr_fit.copy()
    best_seq = list(curr_seqs)
    initial_fit = curr_fit.copy()

    latent_batch_size = gen_batch_size or B
    _, initial_latents = generate_with_raygun(
        curr_seqs,
        raymodel=raymodel,
        esm_model=esm_model,
        alphabet=alphabet,
        proposal_std=0.0,
        device=device,
        batch_size=latent_batch_size,
        shuffle=False,
        return_names=False,
    )
    if len(initial_latents) != B:
        print(f"len(initial_latents)={len(initial_latents)} vs B={B}")
        raise RuntimeError("Failed to obtain initial latent embeddings from Raygun.")

    print(
        f"[SA-BATCH] device={device}"
        + (
            f" ({torch.cuda.get_device_name(torch.cuda.current_device())})"
            if device.type == "cuda"
            else ""
        )
    )
    print(
        f"[SA-BATCH] B={B}, Batch initial min={curr_fit.min():.4f}, max={curr_fit.max():.4f}"
    )

    # ---- temperature schedule ----
    temps = np.geomspace(initial_temp, final_temp, n_steps).astype(float)

    # ---- CSV logger (streaming) ----
    writer = None
    f_csv = None
    log_fields = [
        "restart",
        "step_in_restart",
        "index",
        "current_fitness",
        "current_sequence",
        "current_distance",
    ]
    if step_log_csv:
        os.makedirs(os.path.dirname(step_log_csv) or ".", exist_ok=True)
        f_csv = open(step_log_csv, "w", newline="")
        writer = csv.DictWriter(f_csv, fieldnames=log_fields)
        writer.writeheader()

        start_rows = (
            {
                "restart": 0,
                "step_in_restart": 0,
                "index": i,
                "current_fitness": float(curr_fit[i]),
                "current_sequence": curr_seqs[i],
                "current_distance": 0.0,
            }
            for i in range(B)
        )
        writer.writerows(list(start_rows))
        f_csv.flush()

    # ---- main loop (with optional restarts) ----
    for restart_idx in range(max(1, n_restarts)):

        print(f"[SA-BATCH] restarting annealing pass {restart_idx + 1}/{n_restarts}")
        curr_seqs = list(initial_sequences)
        curr_fit = initial_fit.copy()
        curr_latents = initial_latents.copy()

        restart_start_time = time.perf_counter()
        restart_start_wall = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"[SA-BATCH] restart {restart_idx + 1}/{max(1, n_restarts)} "
            f"begin {restart_start_wall} min_fit={curr_fit.min():.4f} max_fit={curr_fit.max():.4f}"
        )

        for step in range(n_steps):
            T = float(temps[step])
            print(
                f"[SA-BATCH] proposal_std {proposal_std} at temp {T:.6f} (step {step + 1}/{n_steps})"
            )
            prev_curr_seqs = curr_seqs.copy()
            # Propose one new sequence per current sequence
            prop_seqs, prop_latents = generate_with_raygun(
                curr_seqs,
                raymodel=raymodel,
                esm_model=esm_model,
                alphabet=alphabet,
                proposal_std=proposal_std,
                device=device,
                batch_size=(gen_batch_size or B),
                shuffle=False,
            )
            diffs = sum(1 for a, b in zip(curr_seqs, prop_seqs) if a != b)
            print(f"Step {step}: {diffs} / {B} proposals differ from current")

            # Exit if skipped proposals
            if len(prop_latents) != B:
                raise RuntimeError(
                    "Latent encoding count mismatch for proposal sequences."
                )

            # Filter proposals by latent distance
            dists = []
            num_kept = 0

            filtered_props: List[str] = []
            filtered_latents: List[np.ndarray] = []
            for idx, (seq, lat) in enumerate(zip(prop_seqs, prop_latents)):
                dist = _latent_distance(initial_latents[idx], lat)
                dists.append(dist)
                if dist <= latent_distance_cap:
                    filtered_props.append(seq)
                    filtered_latents.append(lat)
                    num_kept += 1
                else:
                    filtered_props.append(curr_seqs[idx])
                    filtered_latents.append(curr_latents[idx])
            print(
                f"  Latent distances: min={min(dists):.4f}, max={max(dists):.4f}, "
                f"mean={np.mean(dists):.4f}, kept={num_kept}/{B} under cap={latent_distance_cap}"
            )

            prop_seqs = filtered_props
            prop_latents = filtered_latents

            # Score proposals
            prop_probs, _, _ = prediction(
                fit_model,
                prop_seqs,
                device=device,
                out_dir=out_dir,
                cache_dir=cache_dir,
                write_to_disk=False,
            )
            prop_fit = np.array(_to_1d_float_list(prop_probs), dtype=float)  # [B]

            # Metropolis accept/reject per index
            delta = prop_fit - curr_fit
            safe_T = max(T, 1e-12)
            accept_prob = np.exp(
                np.clip(delta / safe_T, a_min=-clip_exp, a_max=clip_exp)
            )
            accept_mask = (delta >= 0.0) | (np.random.rand(B) < accept_prob)
            # Add more detailed diagnostics
            print(f"Step {step}:")
            print(f"  Proposals different: {diffs}/{B} ({100*diffs/B:.1f}%)")
            print(
                f"  Delta: min={delta.min():.4f}, max={delta.max():.4f}, mean={delta.mean():.4f}, std={delta.std():.4f}"
            )
            print(
                f"  Accept rate: {accept_mask.sum()}/{B} ({100*accept_mask.mean():.1f}%)"
            )
            print(
                f"  Curr fitness: min={curr_fit.min():.4f}, max={curr_fit.max():.4f}, mean={curr_fit.mean():.4f}"
            )
            print(
                f"  Best fitness: min={best_fit.min():.4f}, max={best_fit.max():.4f}, mean={best_fit.mean():.4f}"
            )
            print(f"  Temperature: {T:.6f}")

            # Update current & best
            for i in range(B):
                if not accept_mask[i]:
                    continue

                # Accept proposal
                curr_fit[i] = prop_fit[i]
                curr_latents[i] = prop_latents[i]
                # Only treat it as a "sequence move" if the sequence changed
                if curr_seqs[i] != prop_seqs[i]:
                    curr_seqs[i] = prop_seqs[i]
                    # Update best if improved
                    if curr_fit[i] > best_fit[i]:
                        best_fit[i] = curr_fit[i]
                        best_seq[i] = curr_seqs[i]
                        if writer is not None:
                            writer.writerow(
                                {
                                    "restart": restart_idx,
                                    "step_in_restart": step + 1,
                                    "index": i,
                                    "current_fitness": float(curr_fit[i]),
                                    "current_sequence": curr_seqs[i],
                                    "current_distance": dists[i],
                                }
                            )
                            f_csv.flush()

            changed = sum(1 for a, b in zip(prev_curr_seqs, curr_seqs) if a != b)
            print(f"  After update: {changed}/{B} current sequences actually changed")

        restart_end_time = time.perf_counter()
        restart_end_wall = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        restart_elapsed = restart_end_time - restart_start_time

        print(
            f"[SA-BATCH] restart {restart_idx + 1}/{max(1, n_restarts)} "
            f"end {restart_end_wall} elapsed={restart_elapsed:.2f}s "
            f"min_fit={curr_fit.min():.4f} max_fit={curr_fit.max():.4f}"
        )
    # ---- final save of best sequences ----

    if f_csv is not None:
        f_csv.close()
        print(f"[SA-BATCH] Step log saved to: {step_log_csv}")


def run_batch(
    model_path: str,
    data_path: str,
    out_dir: str,
    chunk_size: int = 32,
    n_steps: int = 10_000,
    initial_temp: float = 1.0,
    final_temp: float = 1e-3,
    proposal_std: float = 0.1,
    cache_dir: str = "/cluster/tufts/cowenlab/wlou01/modelcache/",
    filenum: Optional[int] = None,
    n_restarts: int = 100,
) -> int:

    # 0) Prepare out dir
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # 1) Load sequences (keep "NA" literal)
    scr_data = pd.read_csv(
        data_path, header=None, names=["sequence"], dtype=str, keep_default_na=False
    )
    seqs = scr_data["sequence"].astype(str).str.strip()
    seqs = seqs[seqs.str.len() > 0].tolist()
    if len(seqs) == 0:
        print("[run_batch] No sequences found after cleaning.")
        return 0

    # 2) Chunk
    seq_batches = [seqs[i : i + chunk_size] for i in range(0, len(seqs), chunk_size)]

    # 3) Init models once
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fit_model = init_model(modelname=model_path, device=device)
    fit_model.eval()

    # Raygun generator + ESM (reused across all batches)
    # raymodel = raygun_4_4mil_800M().to(device).eval()
    localurl = "/cluster/tufts/cowenlab/wlou01/modelcache/rohitsinghlab_raygun_main"
    raymodel, esmdecoder, _ = torch.hub.load(
        localurl, "pretrained_uniref50_4_4mil_800M", source="local"
    )
    raymodel = raymodel.model.to(device).eval()
    esm_model, alphabet = esm2_t33_650M_UR50D()
    esm_model = esm_model.to(device).eval()

    total = 0
    t_total_start = time.perf_counter()

    for bidx, batch in enumerate(tqdm(seq_batches, desc="Running batches")):
        # Clean this batch
        batch_seqs = [s.strip() for s in batch if isinstance(s, str) and s.strip()]
        if not batch_seqs:
            continue

        step_log_csv = f"file{filenum:02d}_b{bidx:05d}_sa_steps.csv"
        step_log_csv = out_root / step_log_csv
        # ---- timing: start
        wall_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        t0 = time.perf_counter()
        print(
            f"[run_batch] batch {bidx:05d} | start {wall_start} | n={len(batch_seqs)} | out={step_log_csv}",
            flush=True,
        )

        # Run batched SA (handles per-step best logging internally)
        protein_design_batch(
            fit_model=fit_model,
            initial_sequences=batch_seqs,
            raymodel=raymodel,
            esm_model=esm_model,
            alphabet=alphabet,
            n_steps=n_steps,
            initial_temp=initial_temp,
            final_temp=final_temp,
            proposal_std=proposal_std,
            device=device,
            out_dir=out_root,
            cache_dir=cache_dir,
            log_every=50,
            step_log_csv=step_log_csv,
            n_restarts=n_restarts,
        )
        # ---- timing: end
        t1 = time.perf_counter()
        elapsed = t1 - t0
        rate = (len(batch_seqs) / elapsed) if elapsed > 0 else float("inf")
        wall_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"[run_batch] batch {bidx:05d} | end {wall_end} | elapsed={elapsed:.2f}s | rate={rate:.2f} seq/s",
            flush=True,
        )

        total += len(batch_seqs)

        # Optional: free cached GPU mem between batches
        if device.type == "cuda":
            torch.cuda.empty_cache()

    total_elapsed = time.perf_counter() - t_total_start
    print(
        f"[run_batch] DONE | batches={len(seq_batches)} | total_seqs={total} | total_time={total_elapsed:.2f}s | overall_rate={(total/total_elapsed) if total_elapsed>0 else float('inf'):.2f} seq/s",
        flush=True,
    )

    return total


import argparse


def main():
    parser = argparse.ArgumentParser(description="Run batched SA over a sequence file.")
    parser.add_argument(
        "--model-path", type=str, default="checkpoints/best_model_v2.pth"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to one-sequence-per-line text file",
    )
    parser.add_argument("--out-dir", type=str, default="outputs_sa")
    parser.add_argument(
        "--fileid",
        type=int,
        default=0,
        help="Optional ID to tag this run (passed to run_batch as filenum)",
    )

    # Pass-through knobs for run_batch / SA
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--n-steps", type=int, default=20)
    parser.add_argument("--initial-temp", type=float, default=1.0)
    parser.add_argument("--final-temp", type=float, default=1e-3)
    parser.add_argument("--proposal-std", type=float, default=0.1)
    parser.add_argument("--n-restarts", type=int, default=100)
    parser.add_argument(
        "--cache-dir", type=str, default="/cluster/tufts/cowenlab/wlou01/modelcache/"
    )

    args = parser.parse_args()

    processed = run_batch(
        model_path=args.model_path,
        data_path=args.data_path,
        out_dir=args.out_dir,
        chunk_size=args.chunk_size,
        n_steps=args.n_steps,
        initial_temp=args.initial_temp,
        final_temp=args.final_temp,
        proposal_std=args.proposal_std,
        cache_dir=args.cache_dir,
        filenum=args.fileid,
        n_restarts=args.n_restarts,
    )
    print("Total sequences processed:", processed)


if __name__ == "__main__":
    main()
