from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Mapping


QUANTIZABLE_MODULE_TYPES = {"Conv2d", "Linear"}


@dataclass
class SD15MappingReport:
    matched_tensor_count: int = 0
    unmatched_tensor_count: int = 0
    quantized_tensor_count: int = 0
    fp16_kept_tensor_count: int = 0
    preserved_non_unet_tensor_count: int = 0
    skipped_tensor_reasons: dict[str, int] = field(default_factory=dict)


@dataclass
class SD15WeightMapping:
    module_to_stripped_weight_key: dict[str, str]
    module_to_source_weight_key: dict[str, str]
    stripped_weight_to_module_name: dict[str, str]
    source_weight_to_module_name: dict[str, str]
    quantizable_module_names: list[str]
    report: SD15MappingReport


def collect_module_types_from_named_modules(
    named_modules: Iterable[tuple[str, object]],
) -> dict[str, str]:
    return {
        name: module.__class__.__name__
        for name, module in named_modules
        if name
    }


def build_sd15_unet_weight_mapping(
    stripped_unet_state_dict: Mapping[str, object],
    live_module_types: Mapping[str, str],
    live_state_keys: Iterable[str] | None = None,
    unet_prefix: str = "model.diffusion_model.",
) -> SD15WeightMapping:
    live_state_key_set = set(live_state_keys or [])
    skipped = Counter()

    module_to_stripped_weight_key: dict[str, str] = {}
    module_to_source_weight_key: dict[str, str] = {}
    stripped_weight_to_module_name: dict[str, str] = {}
    source_weight_to_module_name: dict[str, str] = {}

    matched_tensor_count = 0
    unmatched_tensor_count = 0

    for stripped_key in stripped_unet_state_dict.keys():
        if not stripped_key.endswith(".weight"):
            continue

        module_name = stripped_key[: -len(".weight")]
        module_type = live_module_types.get(module_name)

        if live_state_key_set and stripped_key not in live_state_key_set:
            unmatched_tensor_count += 1
            skipped["no_live_parameter"] += 1
            continue

        if module_type is None:
            unmatched_tensor_count += 1
            skipped["no_live_module"] += 1
            continue

        if module_type not in QUANTIZABLE_MODULE_TYPES:
            unmatched_tensor_count += 1
            skipped["non_quantizable_module"] += 1
            continue

        source_key = f"{unet_prefix}{stripped_key}" if unet_prefix else stripped_key
        matched_tensor_count += 1
        module_to_stripped_weight_key[module_name] = stripped_key
        module_to_source_weight_key[module_name] = source_key
        stripped_weight_to_module_name[stripped_key] = module_name
        source_weight_to_module_name[source_key] = module_name

    for module_name, module_type in live_module_types.items():
        if module_type not in QUANTIZABLE_MODULE_TYPES:
            continue
        weight_key = f"{module_name}.weight"
        if weight_key not in stripped_unet_state_dict:
            skipped["missing_source_weight_for_live_module"] += 1

    report = SD15MappingReport(
        matched_tensor_count=matched_tensor_count,
        unmatched_tensor_count=unmatched_tensor_count,
        skipped_tensor_reasons=dict(sorted(skipped.items())),
    )

    return SD15WeightMapping(
        module_to_stripped_weight_key=module_to_stripped_weight_key,
        module_to_source_weight_key=module_to_source_weight_key,
        stripped_weight_to_module_name=stripped_weight_to_module_name,
        source_weight_to_module_name=source_weight_to_module_name,
        quantizable_module_names=sorted(module_to_stripped_weight_key.keys()),
        report=report,
    )


def finalize_sd15_mapping_report(
    base_report: SD15MappingReport,
    quantized_tensor_count: int,
    fp16_kept_tensor_count: int,
    preserved_non_unet_tensor_count: int = 0,
) -> SD15MappingReport:
    return SD15MappingReport(
        matched_tensor_count=base_report.matched_tensor_count,
        unmatched_tensor_count=base_report.unmatched_tensor_count,
        quantized_tensor_count=quantized_tensor_count,
        fp16_kept_tensor_count=fp16_kept_tensor_count,
        preserved_non_unet_tensor_count=preserved_non_unet_tensor_count,
        skipped_tensor_reasons=dict(base_report.skipped_tensor_reasons),
    )


def format_sd15_mapping_report(report: SD15MappingReport) -> str:
    lines = [
        f"  matched tensor count: {report.matched_tensor_count}",
        f"  unmatched tensor count: {report.unmatched_tensor_count}",
        f"  quantized tensor count: {report.quantized_tensor_count}",
        f"  fp16-kept tensor count: {report.fp16_kept_tensor_count}",
        f"  preserved non-UNet tensor count: {report.preserved_non_unet_tensor_count}",
    ]

    if report.skipped_tensor_reasons:
        lines.append("  skipped tensor reasons:")
        for reason, count in report.skipped_tensor_reasons.items():
            lines.append(f"    - {reason}: {count}")

    return "\n".join(lines)
