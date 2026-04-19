"""
Quantize SD1.x / SD1.5 checkpoints to FP8 with a ComfyUI-first HSWQ V1 path.

This script is intentionally additive and conservative:
- checkpoint ingest uses ComfyUI loading and model detection
- calibration uses ComfyUI text encoding and sampling internals
- only standard-compatible FP8 E4M3FN output is produced
- SDXL and other existing scripts are left untouched
"""

from __future__ import annotations

import argparse
import copy
import gc
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from tqdm import tqdm

from hswq_sd15_comfy_loader import (
    UnsupportedSD15ModelError,
    load_sd15_checkpoint_for_hswq,
)
from hswq_sd15_mapping import (
    build_sd15_unet_weight_mapping,
    collect_module_types_from_named_modules,
    finalize_sd15_mapping_report,
    format_sd15_mapping_report,
)

try:
    from histogram.weighted_histogram_mse_fast import (
        HSWQWeightedHistogramOptimizerFast as HSWQWeightedHistogramOptimizer,
    )
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    histogram_dir = os.path.join(current_dir, "histogram")
    if histogram_dir not in sys.path:
        sys.path.insert(0, histogram_dir)
    from weighted_histogram_mse_fast import (  # type: ignore[no-redef]
        HSWQWeightedHistogramOptimizerFast as HSWQWeightedHistogramOptimizer,
    )


HSWQ_VERSION = "hswq_sd15_v1"
CALIBRATION_CFG = 7.5
CALIBRATION_SAMPLER = "euler"
CALIBRATION_SCHEDULER = "normal"
CALIBRATION_NEGATIVE_PROMPT = ""


class DualMonitor:
    def __init__(self) -> None:
        self.output_sum = 0.0
        self.output_sq_sum = 0.0
        self.count = 0
        self.channel_importance = None

    def update(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor) -> None:
        with torch.no_grad():
            out_detached = output_tensor.detach().float()
            batch_mean = out_detached.mean().item()
            batch_sq_mean = (out_detached ** 2).mean().item()

            self.output_sum += batch_mean
            self.output_sq_sum += batch_sq_mean

            inp_detached = input_tensor.detach()
            if inp_detached.dim() == 4:
                current_imp = inp_detached.abs().mean(dim=(0, 2, 3))
            elif inp_detached.dim() == 3:
                current_imp = inp_detached.abs().mean(dim=(0, 1))
            elif inp_detached.dim() == 2:
                current_imp = inp_detached.abs().mean(dim=0)
            else:
                current_imp = torch.ones(1, device=inp_detached.device, dtype=inp_detached.dtype)

            if self.channel_importance is None:
                self.channel_importance = current_imp
            else:
                self.channel_importance = (
                    self.channel_importance * self.count + current_imp
                ) / (self.count + 1)

            self.count += 1

    def get_sensitivity(self) -> float:
        if self.count == 0:
            return 0.0
        mean = self.output_sum / self.count
        sq_mean = self.output_sq_sum / self.count
        return sq_mean - mean ** 2


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SD1.x / SD1.5 FP8 Quantization (HSWQ V1, ComfyUI-first)"
    )
    parser.add_argument("--input", type=str, required=True, help="Path to an SD1.x checkpoint (.safetensors or .ckpt)")
    parser.add_argument("--output", type=str, required=True, help="Output safetensors path or output stem")
    parser.add_argument("--calib_file", type=str, required=True, help="Calibration prompts text file")
    parser.add_argument("--num_calib_samples", type=int, default=32, help="Number of calibration samples")
    parser.add_argument("--num_inference_steps", type=int, default=25, help="Denoising steps per calibration sample")
    parser.add_argument("--keep_ratio", type=float, default=0.10, help="Ratio of matched layers to keep in FP16")
    parser.add_argument("--latent", type=int, default=128, help="Latent height/width used during calibration")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for deterministic calibration")
    parser.add_argument("--device", type=str, default=None, help="Preferred device (default: auto)")
    parser.add_argument(
        "--output_mode",
        type=str,
        default="unet",
        choices=("unet", "full", "both"),
        help="Write UNet-only output, full-checkpoint output, or both",
    )
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress information")
    return parser


def validate_output_request(output_path: str, output_mode: str) -> None:
    if output_mode in {"unet", "full"} and not output_path.lower().endswith(".safetensors"):
        raise ValueError(
            f"--output must end with '.safetensors' when --output_mode={output_mode}."
        )


def derive_output_paths(output_path: str, output_mode: str) -> dict[str, str]:
    validate_output_request(output_path, output_mode)

    if output_mode != "both":
        return {output_mode: output_path}

    base = output_path
    lower = output_path.lower()
    for suffix in (".safetensors", ".sft", ".ckpt", ".pt", ".bin"):
        if lower.endswith(suffix):
            base = output_path[: -len(suffix)]
            break

    return {
        "unet": f"{base}_unet.safetensors",
        "full": f"{base}_full.safetensors",
    }


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.num_calib_samples <= 0:
        raise ValueError("--num_calib_samples must be greater than 0.")
    if args.num_inference_steps <= 0:
        raise ValueError("--num_inference_steps must be greater than 0.")
    if not 0.0 <= args.keep_ratio <= 1.0:
        raise ValueError("--keep_ratio must be between 0.0 and 1.0.")
    if args.latent <= 0:
        raise ValueError("--latent must be greater than 0.")
    validate_output_request(args.output, args.output_mode)
    return args


def resolve_device(preferred_device: str | None) -> str:
    if preferred_device:
        return preferred_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def clear_torch_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_calibration_prompts(calib_file: str, num_samples: int) -> list[str]:
    with open(calib_file, "r", encoding="utf-8") as handle:
        prompts = [line.strip() for line in handle.readlines() if line.strip()]

    if not prompts:
        raise ValueError(f"No calibration prompts were found in {calib_file}.")

    if len(prompts) < num_samples:
        multiplier = (num_samples // len(prompts)) + 1
        prompts = (prompts * multiplier)[:num_samples]
    else:
        prompts = prompts[:num_samples]
    return prompts


def encode_conditioning(clip, text: str):
    tokens = clip.tokenize(text)
    return clip.encode_from_tokens_scheduled(tokens, show_pbar=False)


def hook_fn(module, inputs, output, name: str, dual_monitors: dict[str, DualMonitor]) -> None:
    if name not in dual_monitors:
        dual_monitors[name] = DualMonitor()
    dual_monitors[name].update(inputs[0], output)


def register_dual_monitor_hooks(live_unet, target_module_names: list[str]):
    target_set = set(target_module_names)
    target_modules = {}
    handles = []
    dual_monitors: dict[str, DualMonitor] = {}

    for name, module in live_unet.named_modules():
        if name not in target_set:
            continue
        target_modules[name] = module
        handle = module.register_forward_hook(
            lambda m, i, o, n=name: hook_fn(m, i, o, n, dual_monitors)
        )
        handles.append(handle)

    return handles, target_modules, dual_monitors


def run_sd15_calibration(
    model_patcher,
    clip,
    latent_size: int,
    prompts: list[str],
    num_inference_steps: int,
    seed: int,
    verbose: bool,
) -> None:
    import comfy.sample as comfy_sample

    negative = encode_conditioning(clip, CALIBRATION_NEGATIVE_PROMPT)
    latent_image = torch.zeros([1, 4, latent_size, latent_size], dtype=torch.float32, device="cpu")

    progress = tqdm(
        range(len(prompts)),
        desc="Calibration",
        disable=False,
    )

    for index in progress:
        prompt = prompts[index]
        sample_seed = seed + index
        positive = encode_conditioning(clip, prompt)
        noise = comfy_sample.prepare_noise(latent_image, sample_seed)

        progress.set_postfix(
            seed=sample_seed,
            image=f"{latent_size * 8}x{latent_size * 8}",
        )

        if verbose:
            print(
                f"[Calibration] sample {index + 1}/{len(prompts)} | seed={sample_seed} | prompt={prompt[:80]}"
            )

        with torch.no_grad():
            comfy_sample.sample(
                model_patcher,
                noise,
                num_inference_steps,
                CALIBRATION_CFG,
                CALIBRATION_SAMPLER,
                CALIBRATION_SCHEDULER,
                positive,
                negative,
                latent_image,
                disable_pbar=not verbose,
                seed=sample_seed,
            )

        if (index + 1) % 4 == 0:
            clear_torch_cache()


def analyze_layer_sensitivity(
    target_module_names: list[str],
    dual_monitors: dict[str, DualMonitor],
    keep_ratio: float,
) -> tuple[list[tuple[str, float]], set[str]]:
    layer_sensitivities = []
    for name in target_module_names:
        if name in dual_monitors:
            layer_sensitivities.append((name, dual_monitors[name].get_sensitivity()))

    layer_sensitivities.sort(key=lambda item: item[1], reverse=True)
    num_keep = int(len(layer_sensitivities) * keep_ratio)
    keep_layers = {name for name, _ in layer_sensitivities[:num_keep]}
    return layer_sensitivities, keep_layers


def compute_weight_amax_dict(
    target_modules: dict[str, torch.nn.Module],
    keep_layers: set[str],
    dual_monitors: dict[str, DualMonitor],
    device: str,
    verbose: bool,
) -> dict[str, float]:
    weight_amax_dict: dict[str, float] = {}
    optimizer = HSWQWeightedHistogramOptimizer(
        bins=4096,
        num_candidates=200,
        refinement_iterations=3,
        device=device,
    )

    iterator = tqdm(
        sorted(target_modules.items()),
        desc="HSWQ analysis",
        disable=not verbose,
    )
    for name, module in iterator:
        if name in keep_layers:
            continue

        importance = None
        if name in dual_monitors:
            importance = dual_monitors[name].channel_importance

        optimal_amax = optimizer.compute_optimal_amax(
            module.weight.data,
            importance,
            scaled=False,
        )
        weight_amax_dict[name] = float(optimal_amax)

        if verbose:
            iterator.set_postfix(module=name[:40], amax=f"{optimal_amax:.6f}")

    return weight_amax_dict


def quantize_tensor(
    tensor: torch.Tensor,
    module_name: str,
    keep_layers: set[str],
    weight_amax_dict: dict[str, float],
    device: str,
) -> torch.Tensor:
    if module_name in keep_layers:
        return tensor.to(torch.float16).cpu().contiguous()

    if module_name not in weight_amax_dict:
        raise RuntimeError(f"Missing amax for quantized module: {module_name}")

    amax = weight_amax_dict[module_name]
    work_tensor = tensor.to(device=device, dtype=torch.float32)
    clamped = torch.clamp(work_tensor, -amax, amax)
    return clamped.to(torch.float8_e4m3fn).cpu().contiguous()


def rewrite_matched_weights(
    base_state_dict: dict[str, torch.Tensor],
    key_to_module_name: dict[str, str],
    keep_layers: set[str],
    weight_amax_dict: dict[str, float],
    device: str,
) -> tuple[dict[str, torch.Tensor], int, int]:
    output_state_dict: dict[str, torch.Tensor] = {}
    quantized_count = 0
    kept_count = 0

    for key, value in base_state_dict.items():
        module_name = key_to_module_name.get(key)
        if module_name is None:
            output_state_dict[key] = value.cpu().contiguous()
            continue

        output_state_dict[key] = quantize_tensor(
            value,
            module_name,
            keep_layers,
            weight_amax_dict,
            device,
        )
        if module_name in keep_layers:
            kept_count += 1
        else:
            quantized_count += 1

    return output_state_dict, quantized_count, kept_count


def build_output_metadata(
    original_metadata: dict[str, str] | None,
    model_family: str,
    args: argparse.Namespace,
    output_mode: str,
) -> dict[str, str]:
    metadata = {}
    if original_metadata:
        metadata.update({str(k): str(v) for k, v in original_metadata.items()})

    metadata.update(
        {
            "model_family": model_family,
            "hswq_version": HSWQ_VERSION,
            "keep_ratio": str(args.keep_ratio),
            "num_calib_samples": str(args.num_calib_samples),
            "num_inference_steps": str(args.num_inference_steps),
            "latent": str(args.latent),
            "output_mode": output_mode,
        }
    )
    return metadata


def save_state_dict(
    state_dict: dict[str, torch.Tensor],
    output_path: str,
    metadata: dict[str, str],
) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cpu_dict = {key: value.cpu().contiguous() for key, value in state_dict.items()}
    save_file(cpu_dict, output_path, metadata=metadata)


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = validate_args(parser.parse_args(argv))

    device = resolve_device(args.device)
    seed_everything(args.seed)

    print("HSWQ SD1.x / SD1.5 V1 (ComfyUI-first)")
    print(f"Device: {device}")
    print(f"Input: {args.input}")
    print(f"Output mode: {args.output_mode}")

    try:
        loaded = load_sd15_checkpoint_for_hswq(
            args.input,
            device=device,
            verbose=args.verbose,
        )
    except UnsupportedSD15ModelError as exc:
        raise SystemExit(f"Unsupported checkpoint: {exc}") from exc

    model_patcher = loaded["model_patcher"]
    clip = loaded["clip"]
    original_state_dict = loaded["original_state_dict"]
    original_metadata = loaded["metadata"]
    family = loaded["family"]
    stripped_unet_state_dict = loaded["stripped_unet_state_dict"]

    live_unet = model_patcher.model.diffusion_model
    live_module_types = collect_module_types_from_named_modules(live_unet.named_modules())
    live_state_keys = set(live_unet.state_dict().keys())

    mapping = build_sd15_unet_weight_mapping(
        stripped_unet_state_dict=stripped_unet_state_dict,
        live_module_types=live_module_types,
        live_state_keys=live_state_keys,
        unet_prefix=family.unet_prefix,
    )

    if not mapping.quantizable_module_names:
        raise SystemExit(
            "No quantizable SD1.x UNet Conv2d/Linear weights were matched. Aborting."
        )

    print("Mapping summary:")
    print(format_sd15_mapping_report(mapping.report))

    prompts = load_calibration_prompts(args.calib_file, args.num_calib_samples)
    print(
        f"Running SD1.x calibration with {len(prompts)} prompts, "
        f"{args.num_inference_steps} denoising steps, latent {args.latent} "
        f"({args.latent * 8}x{args.latent * 8} image space)."
    )

    handles, target_modules, dual_monitors = register_dual_monitor_hooks(
        live_unet,
        mapping.quantizable_module_names,
    )

    try:
        run_sd15_calibration(
            model_patcher=model_patcher,
            clip=clip,
            latent_size=args.latent,
            prompts=prompts,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
            verbose=args.verbose,
        )
    finally:
        for handle in handles:
            handle.remove()

    layer_sensitivities, keep_layers = analyze_layer_sensitivity(
        mapping.quantizable_module_names,
        dual_monitors,
        args.keep_ratio,
    )

    print(f"Matched quantizable layers: {len(mapping.quantizable_module_names)}")
    print(f"FP16-kept layers: {len(keep_layers)} (top {args.keep_ratio * 100:.1f}%)")
    if layer_sensitivities:
        print("Top 5 sensitive layers:")
        for index, (name, sensitivity) in enumerate(layer_sensitivities[:5], start=1):
            print(f"  {index}. {name}: {sensitivity:.6f}")

    print("Computing HSWQ amax values...")
    weight_amax_dict = compute_weight_amax_dict(
        target_modules=target_modules,
        keep_layers=keep_layers,
        dual_monitors=dual_monitors,
        device=device,
        verbose=args.verbose,
    )

    print(
        f"Quantization plan: {len(weight_amax_dict)} FP8 layers, "
        f"{len(keep_layers)} FP16-kept layers."
    )

    output_paths = derive_output_paths(args.output, args.output_mode)
    preserved_non_unet_tensor_count = sum(
        1 for key in original_state_dict if not key.startswith(family.unet_prefix)
    )

    if "unet" in output_paths:
        print(f"Writing UNet-only output: {output_paths['unet']}")
        unet_state_dict, quantized_count, kept_count = rewrite_matched_weights(
            base_state_dict=copy.copy(stripped_unet_state_dict),
            key_to_module_name=mapping.stripped_weight_to_module_name,
            keep_layers=keep_layers,
            weight_amax_dict=weight_amax_dict,
            device=device,
        )
        unet_metadata = build_output_metadata(
            original_metadata,
            family.model_family,
            args,
            "unet",
        )
        save_state_dict(unet_state_dict, output_paths["unet"], unet_metadata)
        unet_report = finalize_sd15_mapping_report(
            mapping.report,
            quantized_tensor_count=quantized_count,
            fp16_kept_tensor_count=kept_count,
            preserved_non_unet_tensor_count=0,
        )
        print("UNet-only output summary:")
        print(format_sd15_mapping_report(unet_report))

    if "full" in output_paths:
        print(f"Writing full-checkpoint output: {output_paths['full']}")
        full_state_dict, quantized_count, kept_count = rewrite_matched_weights(
            base_state_dict=copy.copy(original_state_dict),
            key_to_module_name=mapping.source_weight_to_module_name,
            keep_layers=keep_layers,
            weight_amax_dict=weight_amax_dict,
            device=device,
        )
        full_metadata = build_output_metadata(
            original_metadata,
            family.model_family,
            args,
            "full",
        )
        save_state_dict(full_state_dict, output_paths["full"], full_metadata)
        full_report = finalize_sd15_mapping_report(
            mapping.report,
            quantized_tensor_count=quantized_count,
            fp16_kept_tensor_count=kept_count,
            preserved_non_unet_tensor_count=preserved_non_unet_tensor_count,
        )
        print("Full-checkpoint output summary:")
        print(format_sd15_mapping_report(full_report))

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
