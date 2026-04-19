# How to quantize SD1.5

This is an **experimental, ComfyUI-first, V1-only** path for standard **SD1.x / SD1.5 checkpoints**.

The current goal is conservative and practical:

- input: SD1.x / SD1.5 checkpoint (`.safetensors` or `.ckpt`)
- quantization target: UNet only
- output priority: UNet-only FP8 safetensors for ComfyUI `diffusion_models` testing
- optional output: full safetensors checkpoint with non-UNet weights preserved

## What this supports

- Standard 4-channel SD1.x / SD1.5 checkpoints detected by ComfyUI as `SD15`
- HSWQ V1 only (`torch.float8_e4m3fn`)
- ComfyUI-backed checkpoint loading and Comfy-backed denoising calibration
- `unet`, `full`, and `both` output modes

## What this does not support yet

- SD2.x
- SDXL
- Flux
- Z Image
- inpaint or instruct-pix2pix SD1.x variants
- V2 / scaled output
- guaranteed day-one support for loading the full rewritten checkpoint with stock `CheckpointLoaderSimple`

## Colab install

```bash
pip install git+https://github.com/Apache0ne/Hybrid-Sensitivity-Weighted-Quantization.git
```

This installs the new CLI entry point:

```bash
hswq-quantize-sd15 --help
```

If you are running from a cloned checkout instead of `pip install`, this also works:

```bash
python quantize_sd15_hswq_v1.py --help
```

## UNet-only output

This is the recommended first run.

```bash
python quantize_sd15_hswq_v1.py \
  --input "/content/models/my_sd15_model.safetensors" \
  --output "/content/output/my_sd15_model_hswq_unet.safetensors" \
  --calib_file "sample/calibration_prompts_sd15.txt" \
  --num_calib_samples 32 \
  --num_inference_steps 25 \
  --keep_ratio 0.10 \
  --latent 128 \
  --output_mode unet
```

## Full-checkpoint output

This preserves original non-UNet weights and only rewrites mapped UNet weights.

```bash
python quantize_sd15_hswq_v1.py \
  --input "/content/models/my_sd15_model.safetensors" \
  --output "/content/output/my_sd15_model_hswq_full.safetensors" \
  --calib_file "sample/calibration_prompts_sd15.txt" \
  --num_calib_samples 32 \
  --num_inference_steps 25 \
  --keep_ratio 0.10 \
  --latent 128 \
  --output_mode full
```

## Both outputs

`both` derives two sibling files from the provided output stem:

- `<stem>_unet.safetensors`
- `<stem>_full.safetensors`

```bash
python quantize_sd15_hswq_v1.py \
  --input "/content/models/my_sd15_model.safetensors" \
  --output "/content/output/my_sd15_model_hswq_bundle.safetensors" \
  --calib_file "sample/calibration_prompts_sd15.txt" \
  --num_calib_samples 32 \
  --num_inference_steps 25 \
  --keep_ratio 0.10 \
  --latent 128 \
  --output_mode both
```

## Expected ComfyUI test flow

1. Copy the UNet-only output file into `ComfyUI/models/diffusion_models/`.
2. Load the original SD1.5 CLIP and VAE from the source checkpoint or your usual SD1.5 assets.
3. Load the FP8 UNet with `UNETLoader`.
4. Run a normal SD1.5 text-to-image workflow and compare against the original model.

## Limitations and likely failure modes

- Unsupported-family rejection is strict on purpose. If the script says the checkpoint is not standard SD1.x / SD1.5, it is not attempting a risky fallback.
- CPU mode is expected to be very slow.
- `full` mode writes **safetensors only**, even if the input was `.ckpt`.
- Some third-party checkpoints may have extra or unusual UNet keys that are preserved but not quantized if they do not map cleanly to live ComfyUI `Conv2d` / `Linear` modules.
- The full-checkpoint output is primarily for controlled testing. The UNet-only output is the main supported target in this first pass.
