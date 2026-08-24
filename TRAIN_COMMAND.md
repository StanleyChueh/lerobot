# GR00T N1.7 training commands — openarm pringles V14

Dataset: `ethanCSL/openarm_visuomotor_VR_pringles_V14_background`
(400 eps, 143,646 frames @ 20 fps, 16-dim state/action `LJ1..LJ8,RJ1..RJ8`, 3 cameras)

Notes that apply to both:
- `relative_exclude_joints` must name the real joints (`LJ8`/`RJ8`); `"gripper"` matches
  nothing in this dataset and silently trains both grippers as deltas.
- `chunk_size=40` is N1.7's native action horizon; `n_action_steps=16` replans every 0.8 s.
- `--steps` counts micro-batches, not optimizer updates. Both recipes below give
  20,000 updates at effective batch 32 (~5 epochs).

## RTX 5090 (32 GB) — full action-head fine-tune

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True lerobot-train \
  --dataset.repo_id=ethanCSL/openarm_visuomotor_VR_pringles_V14_background \
  --dataset.image_transforms.enable=true \
  --policy.type=groot \
  --policy.device=cuda \
  --policy.base_model_path=nvidia/GR00T-N1.7-3B \
  --policy.embodiment_tag=new_embodiment \
  --policy.chunk_size=40 \
  --policy.n_action_steps=16 \
  --policy.use_relative_actions=true \
  --policy.relative_exclude_joints='["LJ8","RJ8"]' \
  --policy.use_bf16=true \
  --policy.model_params_fp32=false \
  --policy.push_to_hub=true \
  --policy.repo_id=ethanCSL/openarm_visuomotor_VR_pringles_V14_background_gr00t \
  --seed=42 \
  --batch_size=16 \
  --accelerator.gradient_accumulation.steps=2 \
  --steps=40000 \
  --num_workers=8 \
  --save_checkpoint=true \
  --save_freq=10000 \
  --use_policy_training_preset=true \
  --env_eval_freq=0 \
  --eval_steps=0 \
  --log_freq=50 \
  --output_dir=outputs/trains/openarm_visuomotor_VR_pringles_V14_background_gr00t \
  --job_name=openarm_V14_background_gr00t \
  --wandb.enable=false
```

If it OOMs, keep batch x accum = 32: `--batch_size=12 --accelerator.gradient_accumulation.steps=3 --steps=53000`.

## RTX 5080 (16 GB) — frozen diffusion transformer

Full head is 1.62B trainable params; at 6 bytes/param (grad + 2 Adam moments in bf16)
that is 9.7 GB on top of 6.9 GB of weights = 16.6 GB, over a 15.46 GB card.
`--policy.tune_diffusion_model=false` drops trainable to 0.53B (~10.1 GB fixed).

Same as above but with:

```
  --policy.tune_diffusion_model=false \
  --batch_size=8 \
  --accelerator.gradient_accumulation.steps=4 \
  --steps=80000 \
  --num_workers=6 \
  --save_freq=20000 \
```
