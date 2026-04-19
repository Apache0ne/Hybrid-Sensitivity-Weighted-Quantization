import pytest

from quantize_sd15_hswq_v1 import (
    build_arg_parser,
    derive_output_paths,
    validate_output_request,
)


def test_sd15_cli_defaults():
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--input",
            "input.safetensors",
            "--output",
            "output.safetensors",
            "--calib_file",
            "sample/calibration_prompts_sd15.txt",
        ]
    )

    assert args.num_calib_samples == 32
    assert args.num_inference_steps == 25
    assert args.keep_ratio == 0.10
    assert args.latent == 128
    assert args.seed == 42
    assert args.output_mode == "unet"
    assert args.device is None
    assert args.verbose is False


def test_invalid_output_mode_is_rejected():
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--input",
                "input.safetensors",
                "--output",
                "output.safetensors",
                "--calib_file",
                "sample/calibration_prompts_sd15.txt",
                "--output_mode",
                "bad-mode",
            ]
        )


def test_both_mode_uses_suffix_paths():
    output_paths = derive_output_paths("runs/model_fp8.safetensors", "both")
    assert output_paths["unet"] == "runs/model_fp8_unet.safetensors"
    assert output_paths["full"] == "runs/model_fp8_full.safetensors"


def test_single_output_requires_safetensors():
    with pytest.raises(ValueError):
        validate_output_request("output.ckpt", "unet")

    with pytest.raises(ValueError):
        validate_output_request("output.ckpt", "full")
