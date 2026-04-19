from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import sys


class UnsupportedSD15ModelError(RuntimeError):
    """Raised when the checkpoint is not a supported SD1.x / SD1.5 family."""


@dataclass
class FamilyDetectionResult:
    model_family: str
    model_config_name: str
    unet_prefix: str
    in_channels: int | None
    has_clip: bool
    accepted: bool
    reason: str | None = None


def _local_root() -> Path:
    return Path(__file__).resolve().parent


def ensure_local_comfy_imports() -> None:
    current_dir = _local_root()
    vendored_comfy_root = current_dir / "ComfyUI-master"
    histogram_dir = current_dir / "histogram"

    if vendored_comfy_root.is_dir():
        vendored_str = str(vendored_comfy_root)
        if vendored_str not in sys.path:
            sys.path.insert(0, vendored_str)

    if histogram_dir.is_dir():
        histogram_str = str(histogram_dir)
        if histogram_str not in sys.path:
            sys.path.insert(0, histogram_str)


def _infer_model_family_tag_from_metadata(metadata: dict[str, Any] | None) -> str:
    if not metadata:
        return "sd1x"

    for key, value in metadata.items():
        haystack = f"{key}={value}".lower()
        if "sd1.5" in haystack or "sd15" in haystack or "stable diffusion 1.5" in haystack:
            return "sd15"
    return "sd1x"


def friendly_model_family_label(model_config_name: str) -> str:
    labels = {
        "SD15": "SD1.x / SD1.5",
        "SD20": "SD2.x",
        "SDXL": "SDXL",
        "SDXLRefiner": "SDXL Refiner",
        "Flux": "Flux",
        "SD15_instructpix2pix": "SD1.x instruct-pix2pix",
    }
    return labels.get(model_config_name, model_config_name or "unknown")


def assess_sd15_family(
    model_config_name: str,
    unet_prefix: str,
    in_channels: int | None,
    has_clip: bool,
    metadata: dict[str, Any] | None = None,
) -> FamilyDetectionResult:
    model_family = _infer_model_family_tag_from_metadata(metadata)

    if not has_clip:
        return FamilyDetectionResult(
            model_family=model_family,
            model_config_name=model_config_name,
            unet_prefix=unet_prefix,
            in_channels=in_channels,
            has_clip=has_clip,
            accepted=False,
            reason=(
                "This file does not contain CLIP/text encoder weights. "
                "Diffusion-only UNet files are not supported for the SD1.x checkpoint ingest path."
            ),
        )

    if model_config_name != "SD15":
        family_label = friendly_model_family_label(model_config_name)
        return FamilyDetectionResult(
            model_family=model_family,
            model_config_name=model_config_name,
            unet_prefix=unet_prefix,
            in_channels=in_channels,
            has_clip=has_clip,
            accepted=False,
            reason=(
                f"Unsupported checkpoint family detected: {family_label}. "
                "This script only supports standard SD1.x / SD1.5 checkpoints."
            ),
        )

    if in_channels != 4:
        return FamilyDetectionResult(
            model_family=model_family,
            model_config_name=model_config_name,
            unet_prefix=unet_prefix,
            in_channels=in_channels,
            has_clip=has_clip,
            accepted=False,
            reason=(
                f"Unsupported SD1.x variant detected (in_channels={in_channels}). "
                "Only standard 4-channel SD1.x / SD1.5 checkpoints are supported. "
                "Inpaint and instruct-pix2pix variants are not supported in this first pass."
            ),
        )

    return FamilyDetectionResult(
        model_family=model_family,
        model_config_name=model_config_name,
        unet_prefix=unet_prefix,
        in_channels=in_channels,
        has_clip=has_clip,
        accepted=True,
        reason=None,
    )


def detect_sd_family(
    input_path: str | None = None,
    state_dict: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> FamilyDetectionResult:
    if input_path is None and state_dict is None:
        raise ValueError("Either input_path or state_dict must be provided to detect_sd_family().")

    ensure_local_comfy_imports()

    from comfy import model_detection
    from comfy import utils as comfy_utils

    if state_dict is None:
        state_dict, metadata = comfy_utils.load_torch_file(input_path, return_metadata=True)

    unet_prefix = model_detection.unet_prefix_from_state_dict(state_dict)
    model_config = model_detection.model_config_from_unet(
        state_dict,
        unet_prefix,
        metadata=metadata,
    )

    model_config_name = model_config.__class__.__name__ if model_config is not None else "unknown"
    in_channels = None
    if model_config is not None:
        in_channels = model_config.unet_config.get("in_channels")

    has_clip = any(key.startswith("cond_stage_model.") for key in state_dict.keys())
    return assess_sd15_family(
        model_config_name=model_config_name,
        unet_prefix=unet_prefix,
        in_channels=in_channels,
        has_clip=has_clip,
        metadata=metadata,
    )


def extract_sd15_unet_state_dict(
    state_dict: dict[str, Any],
    unet_prefix: str = "model.diffusion_model.",
) -> dict[str, Any]:
    stripped = {
        key[len(unet_prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(unet_prefix)
    }
    if not stripped:
        raise UnsupportedSD15ModelError(
            f"No UNet tensors were found under prefix '{unet_prefix}'."
        )
    return stripped


def maybe_extract_clip_and_vae_state_dict(
    state_dict: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    clip_state_dict = {
        key[len("cond_stage_model.") :]: value
        for key, value in state_dict.items()
        if key.startswith("cond_stage_model.")
    }
    vae_state_dict = {
        key[len("first_stage_model.") :]: value
        for key, value in state_dict.items()
        if key.startswith("first_stage_model.")
    }
    return clip_state_dict, vae_state_dict


def load_sd15_checkpoint_for_hswq(
    input_path: str,
    device: str | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    ensure_local_comfy_imports()

    import torch
    from comfy import sd as comfy_sd
    from comfy import utils as comfy_utils

    raw_state_dict, metadata = comfy_utils.load_torch_file(input_path, return_metadata=True)
    detection = detect_sd_family(state_dict=raw_state_dict, metadata=metadata)

    if not detection.accepted:
        raise UnsupportedSD15ModelError(detection.reason or "Unsupported SD1.x checkpoint.")

    stripped_unet_state_dict = extract_sd15_unet_state_dict(
        raw_state_dict,
        detection.unet_prefix,
    )
    clip_state_dict, vae_state_dict = maybe_extract_clip_and_vae_state_dict(raw_state_dict)

    model_options: dict[str, Any] = {}
    te_model_options: dict[str, Any] = {}
    if device == "cpu":
        model_options["dtype"] = torch.float32
        te_model_options["dtype"] = torch.float32

    if verbose:
        print(f"[Loader] Detected family: {detection.model_config_name}")
        print(f"[Loader] UNet prefix: {detection.unet_prefix}")
        print(f"[Loader] Stripped UNet tensors: {len(stripped_unet_state_dict)}")

    loadable_state_dict = dict(raw_state_dict)
    model_patcher, clip, _, _ = comfy_sd.load_state_dict_guess_config(
        loadable_state_dict,
        output_vae=False,
        output_clip=True,
        output_clipvision=False,
        embedding_directory=None,
        output_model=True,
        model_options=model_options,
        te_model_options=te_model_options,
        metadata=metadata,
    )

    if model_patcher is None:
        raise UnsupportedSD15ModelError("ComfyUI failed to build the SD1.x diffusion model.")
    if clip is None:
        raise UnsupportedSD15ModelError(
            "ComfyUI did not load a CLIP/text encoder from the checkpoint."
        )

    return {
        "model_patcher": model_patcher,
        "clip": clip,
        "original_state_dict": raw_state_dict,
        "metadata": metadata or {},
        "family": detection,
        "stripped_unet_state_dict": stripped_unet_state_dict,
        "clip_state_dict": clip_state_dict,
        "vae_state_dict": vae_state_dict,
    }
