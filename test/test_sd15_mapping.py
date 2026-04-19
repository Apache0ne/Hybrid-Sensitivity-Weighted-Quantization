from hswq_sd15_comfy_loader import assess_sd15_family
from hswq_sd15_mapping import build_sd15_unet_weight_mapping


def test_sd15_mapping_matches_expected_keys():
    stripped_unet_state_dict = {
        "input_blocks.0.0.weight": object(),
        "time_embed.0.weight": object(),
        "input_blocks.1.0.in_layers.2.weight": object(),
        "input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight": object(),
        "out.2.weight": object(),
        "input_blocks.1.1.norm.weight": object(),
        "unused_branch.weight": object(),
    }

    live_module_types = {
        "input_blocks.0.0": "Conv2d",
        "time_embed.0": "Linear",
        "input_blocks.1.0.in_layers.2": "Conv2d",
        "input_blocks.1.1.transformer_blocks.0.attn1.to_q": "Linear",
        "out.2": "Conv2d",
        "input_blocks.1.1.norm": "GroupNorm",
    }

    live_state_keys = {
        "input_blocks.0.0.weight",
        "time_embed.0.weight",
        "input_blocks.1.0.in_layers.2.weight",
        "input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight",
        "out.2.weight",
        "input_blocks.1.1.norm.weight",
    }

    mapping = build_sd15_unet_weight_mapping(
        stripped_unet_state_dict=stripped_unet_state_dict,
        live_module_types=live_module_types,
        live_state_keys=live_state_keys,
        unet_prefix="model.diffusion_model.",
    )

    assert mapping.module_to_stripped_weight_key["input_blocks.0.0"] == "input_blocks.0.0.weight"
    assert mapping.module_to_source_weight_key["out.2"] == "model.diffusion_model.out.2.weight"
    assert mapping.report.matched_tensor_count == 5
    assert mapping.report.unmatched_tensor_count == 2
    assert mapping.report.skipped_tensor_reasons["non_quantizable_module"] == 1
    assert mapping.report.skipped_tensor_reasons["no_live_parameter"] == 1


def test_unsupported_family_rejection_mock():
    rejected = assess_sd15_family(
        model_config_name="SDXL",
        unet_prefix="model.diffusion_model.",
        in_channels=4,
        has_clip=True,
        metadata=None,
    )

    assert rejected.accepted is False
    assert "SDXL" in rejected.reason


def test_missing_clip_rejection_mock():
    rejected = assess_sd15_family(
        model_config_name="SD15",
        unet_prefix="model.diffusion_model.",
        in_channels=4,
        has_clip=False,
        metadata=None,
    )

    assert rejected.accepted is False
    assert "CLIP/text encoder" in rejected.reason
