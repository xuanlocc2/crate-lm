import argparse
import json
import math
import os
import re
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import tiktoken
import torch

from crate_overcomplete import CRATE, CRATEConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute lightweight CRATE neuron interpretability metrics using the "
            "OpenAI random-only, OpenAI top-and-random, and Anthropic-style "
            "activation-record protocols."
        )
    )
    parser.add_argument("--out_dir", required=True, help="Directory containing ckpt.pt")
    parser.add_argument("--dataset", default="wikitext2", help="Dataset directory under data/")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--ckpt_filename", default="ckpt.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--layers", default="0,1,2,3", help="Comma-separated layer ids or 'all'")
    parser.add_argument("--num_batches", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_features", type=int, default=32, help="Features per layer to score")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--random_k", type=int, default=10)
    parser.add_argument("--quantile_k", type=int, default=5)
    parser.add_argument("--context_radius", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output", default=None, help="Output JSON path")
    return parser.parse_args()


def safe_corrcoef(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or b.size < 2:
        return 0.0
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    value = float(np.corrcoef(a, b)[0, 1])
    if math.isnan(value):
        return 0.0
    return value


def absolute_dev_explained(true_values, pred_values):
    true_values = np.asarray(true_values, dtype=np.float64)
    pred_values = np.asarray(pred_values, dtype=np.float64)
    denom = np.mean(np.abs(true_values))
    if denom < 1e-12:
        return 0.0
    return float(1.0 - np.mean(np.abs(true_values - pred_values)) / denom)


def normalize_activations(values, max_activation):
    values = np.asarray(values, dtype=np.float64)
    if max_activation <= 1e-12:
        return np.zeros_like(values)
    return np.clip(np.floor(10.0 * np.maximum(values, 0.0) / max_activation), 0, 10)


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'-]*")


def normalize_token(token):
    token = token.strip().lower()
    matches = TOKEN_RE.findall(token)
    if not matches:
        return None
    token = matches[0]
    if len(token) < 2:
        return None
    return token


def decode_tokens(enc, token_ids):
    return [enc.decode([int(token_id)]) for token_id in token_ids]


def make_record(flat_tokens, feature_values, center_idx, radius, enc):
    start = max(0, center_idx - radius)
    end = min(len(flat_tokens), center_idx + radius + 1)
    return {
        "tokens": decode_tokens(enc, flat_tokens[start:end]),
        "activations": feature_values[start:end].astype(float).tolist(),
    }


def keyword_explanation(top_records, max_keywords=8):
    weighted_counts = Counter()
    for record in top_records:
        activations = record["activations"]
        if not activations:
            continue
        local_max = max(activations)
        if local_max <= 0:
            continue
        threshold = 0.5 * local_max
        for token, activation in zip(record["tokens"], activations):
            if activation < threshold:
                continue
            normalized = normalize_token(token)
            if normalized is not None:
                weighted_counts[normalized] += float(activation)
    keywords = [token for token, _ in weighted_counts.most_common(max_keywords)]
    return keywords


def simulate_records(records, keywords, max_activation):
    keyword_set = set(keywords)
    true_values = []
    pred_values = []
    for record in records:
        true_norm = normalize_activations(record["activations"], max_activation)
        for token, true_activation in zip(record["tokens"], true_norm):
            normalized = normalize_token(token)
            pred_activation = 10.0 if normalized in keyword_set else 0.0
            true_values.append(float(true_activation))
            pred_values.append(pred_activation)
    return true_values, pred_values


def score_records(records, keywords, max_activation):
    true_values, pred_values = simulate_records(records, keywords, max_activation)
    return {
        "correlation": safe_corrcoef(true_values, pred_values),
        "absolute_dev_explained": absolute_dev_explained(true_values, pred_values),
        "num_tokens_scored": len(true_values),
    }


def choose_feature_ids(flat_acts, num_features):
    max_values = torch.amax(flat_acts, dim=0)
    nonzero = torch.count_nonzero(flat_acts > 0, dim=0)
    variances = torch.var(flat_acts, dim=0)
    usable = torch.where(nonzero > 0)[0]
    if usable.numel() == 0:
        return []
    score = max_values[usable] * torch.sqrt(variances[usable] + 1e-12)
    k = min(num_features, usable.numel())
    selected = usable[torch.topk(score, k=k).indices]
    return selected.cpu().tolist()


def load_model(args):
    device_type = "cuda" if "cuda" in args.device else "cpu"
    if device_type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Use --device=cpu or activate CUDA venv.")

    ckpt_path = os.path.join(args.out_dir, args.ckpt_filename)
    checkpoint = torch.load(ckpt_path, map_location=args.device)
    model_args = checkpoint["model_args"]
    model = CRATE(CRATEConfig(**model_args))

    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()
    return model, model_args, checkpoint


def load_token_batches(args, block_size):
    rng = np.random.default_rng(args.seed)
    data_path = os.path.join("data", args.dataset, f"{args.split}.bin")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing dataset split: {data_path}")
    data = np.memmap(data_path, dtype=np.uint16, mode="r")
    if len(data) <= block_size + 1:
        raise ValueError(f"{data_path} is too small for block_size={block_size}")

    batches = []
    for _ in range(args.num_batches):
        starts = rng.integers(0, len(data) - block_size - 1, size=args.batch_size)
        batch = np.stack([np.asarray(data[i:i + block_size], dtype=np.int64) for i in starts])
        batches.append(torch.from_numpy(batch))
    return batches


def collect_layer_activations(model, layer_id, batches, args):
    captured = []

    def hook(_module, _inputs, output):
        captured.append(output.detach().cpu())

    handle = model.transformer.h[layer_id].ista.relu.register_forward_hook(hook)
    tokens = []
    device_type = "cuda" if "cuda" in args.device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    try:
        with torch.no_grad():
            for batch in batches:
                tokens.append(batch.reshape(-1).cpu())
                batch = batch.to(args.device)
                with ctx:
                    model(batch)
    finally:
        handle.remove()

    acts = torch.cat(captured, dim=0).reshape(-1, captured[0].shape[-1])
    flat_tokens = torch.cat(tokens, dim=0).numpy()
    return flat_tokens, acts


def score_feature(flat_tokens, flat_acts, feature_id, args, enc):
    values = flat_acts[:, feature_id].cpu().numpy()
    max_activation = float(np.max(values))
    if max_activation <= 0:
        return None

    ordered = np.argsort(values)[::-1]
    top_idx = ordered[: args.top_k]

    rng = np.random.default_rng(args.seed + feature_id)
    random_idx = rng.choice(len(values), size=min(args.random_k, len(values)), replace=False)

    quantile_records = []
    quantiles = [0.0, 0.5, 0.9, 0.99, 0.999, 1.0]
    ascending = np.argsort(values)
    for low, high in zip(quantiles[:-1], quantiles[1:]):
        start = int(low * len(ascending))
        end = max(start + 1, int(high * len(ascending)))
        candidates = ascending[start:end]
        if len(candidates) == 0:
            continue
        sample_size = min(args.quantile_k, len(candidates))
        chosen = rng.choice(candidates, size=sample_size, replace=False)
        for idx in chosen:
            quantile_records.append(make_record(flat_tokens, values, int(idx), args.context_radius, enc))

    top_records = [make_record(flat_tokens, values, int(idx), args.context_radius, enc) for idx in top_idx]
    random_records = [make_record(flat_tokens, values, int(idx), args.context_radius, enc) for idx in random_idx]
    keywords = keyword_explanation(top_records)

    random_score = score_records(random_records, keywords, max_activation)
    top_and_random_score = score_records(top_records + random_records, keywords, max_activation)
    anthropic_score = score_records(quantile_records, keywords, max_activation)

    return {
        "feature_id": int(feature_id),
        "max_activation": max_activation,
        "mean_activation": float(np.mean(values)),
        "nonzero_fraction": float(np.mean(values > 0)),
        "explanation_keywords": keywords,
        "metrics": {
            "random": random_score,
            "top_and_random": top_and_random_score,
            "anthropic": anthropic_score,
        },
    }


def assign_feature_to_subspace(ista_weight, feature_id, n_head):
    """Assign an ISTA feature to U_k by dictionary-atom energy."""
    atom = ista_weight[feature_id].detach().float().cpu()
    if atom.numel() % n_head != 0:
        raise ValueError(f"Embedding dim {atom.numel()} is not divisible by n_head={n_head}")
    chunks = atom.reshape(n_head, atom.numel() // n_head)
    energies = torch.sum(chunks * chunks, dim=1)
    total = float(torch.sum(energies).item())
    if total <= 1e-12:
        subspace_id = 0
        energy_ratio = 0.0
    else:
        subspace_id = int(torch.argmax(energies).item())
        energy_ratio = float((energies[subspace_id] / torch.sum(energies)).item())
    return subspace_id, energy_ratio, [float(x) for x in energies.tolist()]


def parse_layers(layers_arg, n_layer):
    if layers_arg == "all":
        return list(range(n_layer))
    layers = [int(x.strip()) for x in layers_arg.split(",") if x.strip()]
    for layer in layers:
        if layer < 0 or layer >= n_layer:
            raise ValueError(f"Layer {layer} outside model range [0, {n_layer - 1}]")
    return layers


def aggregate(results):
    metric_names = ["random", "top_and_random", "anthropic"]
    summary = {}
    for metric_name in metric_names:
        values = [
            feature["metrics"][metric_name]["correlation"]
            for layer in results["layers"]
            for feature in layer["features"]
        ]
        summary[metric_name] = {
            "mean_correlation": float(np.mean(values)) if values else 0.0,
            "std_correlation": float(np.std(values)) if values else 0.0,
            "num_features": len(values),
        }
    return summary


def aggregate_by_subspace(results):
    metric_names = ["random", "top_and_random", "anthropic"]
    grouped = {}
    for layer in results["layers"]:
        for feature in layer["features"]:
            subspace_id = feature.get("subspace_id")
            if subspace_id is None:
                continue
            grouped.setdefault(str(subspace_id), {metric_name: [] for metric_name in metric_names})
            for metric_name in metric_names:
                grouped[str(subspace_id)][metric_name].append(
                    feature["metrics"][metric_name]["correlation"]
                )

    summary = {}
    for subspace_id, metrics in grouped.items():
        summary[subspace_id] = {}
        for metric_name, values in metrics.items():
            summary[subspace_id][metric_name] = {
                "mean_correlation": float(np.mean(values)) if values else 0.0,
                "std_correlation": float(np.std(values)) if values else 0.0,
                "num_features": len(values),
            }
    return summary


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model, model_args, checkpoint = load_model(args)
    layers = parse_layers(args.layers, model_args["n_layer"])
    batches = load_token_batches(args, model_args["block_size"])
    enc = tiktoken.get_encoding("gpt2")

    results = {
        "note": (
            "These are lightweight local proxy scores using the same activation-record "
            "protocol names as the paper. They do not use an LLM explainer/simulator, "
            "so they are not directly comparable to the paper's official scores."
        ),
        "checkpoint": {
            "out_dir": args.out_dir,
            "ckpt_filename": args.ckpt_filename,
            "iter_num": int(checkpoint.get("iter_num", -1)),
            "best_val_loss": float(checkpoint.get("best_val_loss", 0.0)),
        },
        "dataset": {"name": args.dataset, "split": args.split},
        "config": vars(args),
        "model_args": model_args,
        "layers": [],
    }

    for layer_id in layers:
        print(f"Collecting activations for layer {layer_id}...", flush=True)
        flat_tokens, flat_acts = collect_layer_activations(model, layer_id, batches, args)
        feature_ids = choose_feature_ids(flat_acts, args.num_features)
        layer_result = {"layer_id": layer_id, "num_scored_features": len(feature_ids), "features": []}
        ista_weight = model.transformer.h[layer_id].ista.weight

        for index, feature_id in enumerate(feature_ids, start=1):
            feature_result = score_feature(flat_tokens, flat_acts, feature_id, args, enc)
            if feature_result is not None:
                subspace_id, energy_ratio, subspace_energies = assign_feature_to_subspace(
                    ista_weight, feature_id, model_args["n_head"]
                )
                feature_result["subspace_id"] = subspace_id
                feature_result["subspace_energy_ratio"] = energy_ratio
                feature_result["subspace_energies"] = subspace_energies
                layer_result["features"].append(feature_result)
            if index % 8 == 0 or index == len(feature_ids):
                print(f"  layer {layer_id}: scored {index}/{len(feature_ids)} features", flush=True)

        results["layers"].append(layer_result)

    results["summary"] = aggregate(results)
    results["subspace_summary"] = aggregate_by_subspace(results)

    output = args.output
    if output is None:
        safe_name = Path(args.out_dir).name
        output = os.path.join("interpretability_results", f"{safe_name}_{args.dataset}_{args.split}.json")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("")
    print(f"Saved: {output}")
    print("Summary correlation scores:")
    for metric_name, metric_summary in results["summary"].items():
        print(
            f"  {metric_name}: mean={metric_summary['mean_correlation']:.4f}, "
            f"std={metric_summary['std_correlation']:.4f}, "
            f"n={metric_summary['num_features']}"
        )
    print("Subspace U_k summary correlation scores:")
    for subspace_id in sorted(results["subspace_summary"], key=lambda x: int(x)):
        print(f"  U_{subspace_id}:")
        for metric_name, metric_summary in results["subspace_summary"][subspace_id].items():
            print(
                f"    {metric_name}: mean={metric_summary['mean_correlation']:.4f}, "
                f"std={metric_summary['std_correlation']:.4f}, "
                f"n={metric_summary['num_features']}"
            )


if __name__ == "__main__":
    main()
