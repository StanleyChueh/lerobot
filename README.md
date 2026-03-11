<p align="center">
  <img alt="LeRobot, Hugging Face Robotics Library" src="https://cdn-uploads.huggingface.co/production/uploads/631ce4b244503b72277fc89f/MNkMdnJqyPvOAEg20Mafg.png" width="100%">
  <br/>
  <br/>
</p>

<div align="center">

[![Tests](https://github.com/huggingface/lerobot/actions/workflows/nightly.yml/badge.svg?branch=main)](https://github.com/huggingface/lerobot/actions/workflows/nightly.yml?query=branch%3Amain)
[![Python versions](https://img.shields.io/pypi/pyversions/lerobot)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/huggingface/lerobot/blob/main/LICENSE)
[![Status](https://img.shields.io/pypi/status/lerobot)](https://pypi.org/project/lerobot/)
[![Version](https://img.shields.io/pypi/v/lerobot)](https://pypi.org/project/lerobot/)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v2.1-ff69b4.svg)](https://github.com/huggingface/lerobot/blob/main/CODE_OF_CONDUCT.md)

<!-- [![Coverage](https://codecov.io/gh/huggingface/lerobot/branch/main/graph/badge.svg?token=TODO)](https://codecov.io/gh/huggingface/lerobot) -->

</div>

<br/>

<h2 align="center">
    <p><a href="https://huggingface.co/docs/lerobot/so101">
        Test your robot with LeRobot!</a></p>
</h2>

Activate conda env
```bash
cd ~/CSL/lerobot_nn/
conda activate lerobot
```

### Koch Robot:

#### TODO:

In this dev branch, we use the latest LeRobot code to test performance

Code version:commit- **6d0d65a** -2025-12-28

0. Test New SmolVLA ✅

1. Test GR00T N1.5 ✅

2. Test PI0.5 ⏹️

3. Test XVLA ⏹️

4. Attention heat map visualization with SmolVLA and GR00T N1.5(On going...)

#### SmolVLA

Record

Right key to save episode, left key to discard episode

```bash
lerobot-record     --robot.type=koch_follower     --robot.port=/dev/ttyUSB_follower     --robot.id=my_awesome_follower_arm     --robot.cameras="{ front: {type: opencv, index_or_path: /dev/video6, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}}"     --teleop.type=koch_leader     --teleop.port=/dev/ttyUSB_leader     --teleop.id=my_awesome_leader_arm         --dataset.repo_id=ethanCSL/Stanley_grip_block_2color     --dataset.num_episodes=25          --dataset.episode_time_s=30     --dataset.reset_time_s=5     --dataset.single_task="Put the green cube in the box."
```
> **Note**
> Remember to check camera id before recording, use the following command to check camera id, and use ffplay to test it!
> ```
> ls /dev/ttyvideo*
> ```
>
> ```
> ffplay /dev/video*
> ```

resume recording
```bash
lerobot-record     --robot.type=koch_follower     --robot.port=/dev/ttyUSB_follower     --robot.id=my_awesome_follower_arm     --robot.cameras="{ front: {type: opencv, index_or_path: /dev/video6, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}}"     --teleop.type=koch_leader     --teleop.port=/dev/ttyUSB_leader     --teleop.id=my_awesome_leader_arm         --dataset.repo_id=ethanCSL/Stanley_grip_block_2color     --dataset.num_episodes=25          --dataset.episode_time_s=30     --dataset.reset_time_s=5     --dataset.single_task="Put the green cube in the box." --resume=True
```

> **Note**
> Add --resume=true flag to resume recording
> ```
> --resume=True
> ```

Train

If dataset has only two cameras, set one to empty, and remap to fit the format of smolvla_base in dev branch

```
 lerobot-train   --policy.path=lerobot/smolvla_base   --dataset.repo_id=ethanCSL/Stanley_grip_block_2color   --batch_size=16   --steps=20000   --output_dir=outputs/train/Stanley_grip_block_2color   --job_name=my_smolvla_training   --policy.device=cuda   --policy.repo_id=ethanCSL/Stanley_grip_block_2color   --wandb.enable=true   --rename_map='{
    "observation.images.front": "observation.images.camera1",
    "observation.images.top":   "observation.images.camera2"
  }'   --policy.empty_cameras=1 --dataset.video_backend=pyav
```

> **Note**
> Use remap flag,and add
> ```
>  --policy.empty_cameras=1
> ```

Resume training:

```
 lerobot-train   --policy.path=lerobot/smolvla_base   --dataset.repo_id=ethanCSL/Stanley_grip_block_2color   --batch_size=16   --steps=20000   --output_dir=outputs/train/Stanley_grip_block_2color   --job_name=my_smolvla_training   --policy.device=cuda   --policy.repo_id=ethanCSL/Stanley_grip_block_2color   --wandb.enable=true   --rename_map='{
    "observation.images.front": "observation.images.camera1",
    "observation.images.top":   "observation.images.camera2"
  }'   --policy.empty_cameras=1 --resume=true --config_path=/home/bruce/CSL/lerobot_nn/outputs/train/Stanley_grip_block_2color/checkpoints/020000/pretrained_model/train_config.json --dataset.video_backend=pyav
```

> **Note**
> Add
> ```
> --resume=true
> ```
> and
> ```
> --config_path=
> ```

Domain randomization:

It will randomize hue,saturation,constract, brightness,affine,sharpness in training stage for better handling color sensitive task like color-picing task!!!

```
 lerobot-train   --policy.path=lerobot/smolvla_base   --dataset.repo_id=ethanCSL/Stanley_grip_block_2color   --batch_size=16   --steps=20000   --output_dir=outputs/train/Stanley_grip_block_2color   --job_name=my_smolvla_training   --policy.device=cuda   --policy.repo_id=ethanCSL/Stanley_grip_block_2color   --wandb.enable=true   --rename_map='{
    "observation.images.front": "observation.images.camera1",
    "observation.images.top":   "observation.images.camera2"
  }'   --policy.empty_cameras=1 --dataset.image_transforms.enable=true --dataset.image_transforms.random_order=true --dataset.image_transforms.max_num_transforms=6 --dataset.video_backend=pyav
```

Evaluation:

```bash
lerobot-record   --robot.type=koch_follower   --robot.port=/dev/ttyUSB_follower   --robot.id=my_awesome_follower_arm   --robot.cameras='{
    camera1: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30},
    camera2: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}
  }'   --dataset.single_task="Put the green cube in the box."   --dataset.repo_id=ethanCSL/eval_Ting_grip_block   --dataset.episode_time_s=500000   --dataset.num_episodes=10   --teleop.type=koch_leader   --teleop.port=/dev/ttyUSB_leader   --teleop.id=my_awesome_leader_arm   --policy.path=/home/bruce/CSL/lerobot_nn/outputs/train/Stanley_grip_block_2color/checkpoints/020000/pretrained_model   --policy.empty_cameras=1 --dataset.reset_time_s=5  
```

> **Note**
> Remember to add --remap flag,if you used in training

#### GR00T N1.5

##### Record

```bash
lerobot-record     --robot.type=koch_follower     --robot.port=/dev/ttyUSB_follower     --robot.id=my_awesome_follower_arm     --robot.cameras="{ front: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: /dev/video6, width: 640, height: 480, fps: 30}}"     --teleop.type=koch_leader     --teleop.port=/dev/ttyUSB_leader     --teleop.id=my_awesome_leader_arm     --display_data=true     --dataset.repo_id=ethanCSL/multi_block_picking_new_lerobot_gr00t     --dataset.num_episodes=25          --dataset.episode_time_s=30     --dataset.reset_time_s=5     --dataset.single_task="pick up the green block and put into the box"
```

> **Note**
> You don't have to specify policy type in recording, this is just for separating different type of datasets

##### Train

Single GPU training

Train only projector and encoder(Action head DiT and Eagle-2(GR00T N1.5 VLM) are frozen), poor in new robot and new environment

```bash
CUDA_VISIBLE_DEVICES=0  accelerate launch   $(which lerobot-train)   --output_dir=outputs/train_groot_test   --save_checkpoint=true   --batch_size=16   --steps=20000   --save_freq=20000   --log_freq=200   --policy.type=groot   --policy.repo_id=multi_block_picking_new_lerobot_gr00t   --policy.tune_diffusion_model=false   --dataset.repo_id=ethanCSL/multi_block_picking_new_lerobot_gr00t   --dataset.video_backend=pyav   --wandb.enable=false   --wandb.disable_artifact=true   --job_name=groot
```
> **Note**
> RTX5090 LeRobot installation solution
> 
> https://docs.google.com/document/d/1a7i0UfWbSUTbJk_9MFXW-8Dd742hih3A2z61CJjXPG4/edit?usp=sharing
>
> With this command, it needs at least 20GB GPU VRAM to start it, it still needs ~12GB of VRAM to run if lower batch size to 2 or 4

For user with no GPU usage limitation 🙋:
> **Note**
> Tune Action head DiT(Eagle-2 VLM frozen) it can learn new robot better.
> It will need around 27 GB of GPU VRAM
> ```
>  CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 $(which lerobot-train)   --output_dir=outputs/Ting_grip_block_2color_new_gr00t_unfrozen_DiT   --save_checkpoint=true   --batch_size=32   --steps=20000   --save_freq=5000   --log_freq=200   --policy.type=groot   --policy.repo_id=Ting_grip_block_2color_new_gr00t_unfrozen_DiT   --policy.tune_diffusion_model=true   --policy.tune_visual=false   --policy.tune_llm=false   --dataset.repo_id=ethanCSL/Ting_grip_block_2color_new   --dataset.video_backend=pyav   --wandb.enable=false   --wandb.disable_artifact=true   --job_name=groot_tuned
> ```


> Tune diffusion model, Eagle-2 VLM, increase batch size, for 
> It will need around 84 GB of GPU VRAM 🥶
```
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 $(which lerobot-train) \
  --output_dir=outputs/train_groot_unfrozen \
  --save_checkpoint=true \
  --batch_size=32 \
  --steps=20000 \
  --save_freq=5000 \
  --log_freq=200 \
  --policy.type=groot \
  --policy.repo_id=multi_block_picking_new_lerobot_gr00t_unfrozen \
  --policy.tune_diffusion_model=true \
  --policy.tune_visual=true \
  --policy.tune_llm=true \
  --dataset.repo_id=ethanCSL/multi_block_picking_new_lerobot_gr00t \
  --dataset.video_backend=pyav \
  --wandb.enable=false \
  --wandb.disable_artifact=true \
  --job_name=groot_tuned

```

### Async inference

The following command is for GR00T N1.5 async inference, but it can sure apply on all policy😄

Client:
```
python -m lerobot.async_inference.robot_client   --robot.type=koch_follower   --robot.id=my_awesome_follower_arm   --robot.port=/dev/ttyUSB_follower   --robot.cameras="{ front: {type: opencv, index_or_path: /dev/video6, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30} }"   --task="Put the red cube in the box."   --server_address=10.100.4.125:8080   --policy_type=groot   --pretrained_name_or_path=ethanCSL/Stanley_grip_block_2color_gr00t_DiT_unfrozen   --policy_device=cuda   --client_device=cuda   --actions_per_chunk=50     
```

Server:

```
python -m lerobot.async_inference.policy_server   --host=0.0.0.0   --port=8080   --fps=30   --inference_latency=0.033   --obs_queue_timeout=1
```

#### Model evaluation:
##### SmolVLA

Task1: Cube picking task with color-based prompt by SmolVLA

Dataset(100+60=160 episodes):

<img width="1172" height="622" alt="image" src="https://github.com/user-attachments/assets/830ee9d3-c80f-4fae-ba39-c608048c2027" />
<img width="1162" height="633" alt="image" src="https://github.com/user-attachments/assets/429517a2-f804-4249-8187-27b02f2c51ba" />

Dataset link: https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2FethanCSL%2FStanley_grip_block_2color_resume_30_30%2Fepisode_0

Green cube picking
```bash
 lerobot-record   --robot.type=koch_follower   --robot.port=/dev/ttyUSB_follower   --robot.id=my_awesome_follower_arm   --robot.cameras='{
    camera1: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30},
    camera2: {type: opencv, index_or_path: 9, width: 640, height: 480, fps: 30}
  }'   --dataset.single_task="Put the green cube in the box."   --dataset.repo_id=ethanCSL/eval_Ting_grip_block   --dataset.episode_time_s=500000   --dataset.num_episodes=10   --teleop.type=koch_leader   --teleop.port=/dev/ttyUSB_leader   --teleop.id=my_awesome_leader_arm   --policy.path=/home/bruce/CSL/lerobot_nn/outputs/train/Stanley_grip_block_2color_resume_30_30_domain_randomization/checkpoints/040000/pretrained_model   --policy.empty_cameras=1 --dataset.reset_time_s=5 --display_data=True
```

Red cube picking

```bash
 lerobot-record   --robot.type=koch_follower   --robot.port=/dev/ttyUSB_follower   --robot.id=my_awesome_follower_arm   --robot.cameras='{
    camera1: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30},
    camera2: {type: opencv, index_or_path: 9, width: 640, height: 480, fps: 30}
  }'   --dataset.single_task="Put the red cube in the box."   --dataset.repo_id=ethanCSL/eval_Ting_grip_block   --dataset.episode_time_s=500000   --dataset.num_episodes=10   --teleop.type=koch_leader   --teleop.port=/dev/ttyUSB_leader   --teleop.id=my_awesome_leader_arm   --policy.path=/home/bruce/CSL/lerobot_nn/outputs/train/Stanley_grip_block_2color_resume_30_30_domain_randomization/checkpoints/040000/pretrained_model   --policy.empty_cameras=1 --dataset.reset_time_s=5 --display_data=True
```

> **Note**
> Feel free to change the prompt during evaluation, you can test with several prompts to see the difference😎

Task2: Cube sorting task with only one prompt(SmolVLA-paper alike method)

Dataset(100 episodes):

<img width="1152" height="605" alt="image" src="https://github.com/user-attachments/assets/23c3569d-c4ed-4684-a630-9dbcdd396841" />

Dataset link: https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2FethanCSL%2Fsvla_koch_sorting%2Fepisode_0%3Ft%3D2

Sorting green cube,red cube in right and left box.

```bash
 lerobot-record   --robot.type=koch_follower   --robot.port=/dev/ttyUSB_follower   --robot.id=my_awesome_follower_arm   --robot.cameras='{
    camera1: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30},
    camera2: {type: opencv, index_or_path: 8, width: 640, height: 480, fps: 30}
  }'   --dataset.single_task="put the red cube in the right box,green cube in the left box."   --dataset.repo_id=ethanCSL/eval_Ting_grip_block   --dataset.episode_time_s=500000   --dataset.num_episodes=10   --teleop.type=koch_leader   --teleop.port=/dev/ttyUSB_leader   --teleop.id=my_awesome_leader_arm   --policy.path=ethanCSL/svla_koch_sorting_n_stacking   --policy.empty_cameras=1 --dataset.reset_time_s=5
```

Sorting green cube,red cube in right and left box,sorting screw,nut

```
lerobot-record   --robot.type=koch_follower   --robot.port=/dev/ttyUSB_follower   --robot.id=my_awesome_follower_arm   --robot.cameras='{
    camera1: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}, 
    camera2: {type: opencv, index_or_path: 8, width: 640, height: 480, fps: 30}
  }'   --dataset.single_task="put the red cube in the right box,green cube in the left box."   --dataset.repo_id=ethanCSL/eval_Ting_grip_block   --dataset.episode_time_s=500000   --dataset.num_episodes=10   --teleop.type=koch_leader   --teleop.port=/dev/ttyUSB_leader   --teleop.id=my_awesome_leader_arm   --policy.path=ethanCSL/svla_koch_sorting_n_stacking_screw_nut_resume_50   --policy.empty_cameras=1 --dataset.reset_time_s=5
```

Attention heat map visualization

Attention Visualization

SmolVLA

Self attention

```
python src/lerobot/scripts/record_attention_plot_smolvlm_stanley.py     --repo_id "ethanCSL/svla_koch_sorting_n_stacking"          --episode 0     --prompt "Put the red cube in the right box,green cube in the left box." --use_state
```

Cross attention

```
python src/lerobot/scripts/record_attention_plot_cross_stanley.py     --repo_id "ethanCSL/svla_koch_sorting_n_stacking"     --ckpt "ethanCSL/svla_koch_sorting_n_stacking"     --episode 0     --prompt "Put the red cube in the right box,the green cube in the left box." --rename_map='{                                                      
    "observation.images.front": "observation.images.camera1",
    "observation.images.top":   "observation.images.camera2"
}' 
```

#### PI0.5

Training(RTX 5090 is not enough!!)

```
lerobot-train   --dataset.repo_id=ethanCSL/svla_koch_sorting_n_stacking   --policy.type=pi05   --output_dir=./outputs/train/pi05_koch_sorting_n_stacking   --job_name=pi05_sorting_stacking   --policy.repo_id=ethanCSL/pi05_koch_sorting_n_stacking   --policy.pretrained_path=lerobot/pi05_base   --policy.compile_model=true   --policy.gradient_checkpointing=true   --wandb.enable=true   --policy.dtype=bfloat16   --policy.freeze_vision_encoder=false   --policy.train_expert_only=false   --policy.device=cuda   --batch_size=8   --steps=30000 --dataset.video_backend=pyav

```
