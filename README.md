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

Record with three cameras

```
lerobot-record     --robot.type=koch_follower     --robot.port=/dev/ttyUSB_follower     --robot.id=my_awesome_follower_arm     --robot.cameras="{
  front: {type: opencv, index_or_path: 5, width: 640, height: 480, fps: 30, fourcc: MJPG},
  wrist:   {type: opencv, index_or_path: 8, width: 640, height: 480, fps: 30, fourcc: MJPG},
  side:  {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30, fourcc: MJPG}
}"    --teleop.type=koch_leader     --teleop.port=/dev/ttyUSB_leader     --teleop.id=my_awesome_leader_arm         --dataset.repo_id=ethanCSL/svla_koch_sorting_n_stacking_side_front_wrist     --dataset.num_episodes=25          --dataset.episode_time_s=30     --dataset.reset_time_s=5     --dataset.single_task="Put the red cube in the right box,and green cube in the left box." --display_data=true
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

Train with three cameras

```
 lerobot-train   --policy.path=lerobot/smolvla_base   --dataset.repo_id=ethanCSL/svla_koch_sorting_n_stacking_side_front_wrist   --batch_size=16   --steps=40000   --output_dir=outputs/train/svla_koch_sorting_n_stacking_side_front_wrist   --job_name=my_smolvla_training   --policy.device=cuda   --policy.repo_id=ethanCSL/svla_koch_sorting_n_stacking_side_front_wrist   --wandb.enable=false   --rename_map='{
    "observation.images.front": "observation.images.camera1",
    "observation.images.wrist":   "observation.images.camera2",
    "observation.images.side":  "observation.images.camera3"
  }'   --dataset.video_backend=pyav
```

If dataset has only two cameras, set one to empty, and remap to fit the format of smolvla_base in dev branch

```
lerobot-train   --policy.path=lerobot/smolvla_base   --dataset.repo_id=ethanCSL/svla_koch_sorting_n_stacking_wrist_camera   --batch_size=16   --steps=40000   --output_dir=outputs/train/svla_koch_sorting_n_stacking_wrist_camera   --job_name=my_smolvla_training   --policy.device=cuda   --policy.repo_id=ethanCSL/svla_koch_sorting_n_stacking_wrist_camera   --wandb.enable=false  --rename_map='{                                              
    "observation.images.front": "observation.images.camera1",
    "observation.images.wrist":   "observation.images.camera2"
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

Unfrozen Vision encoder(SigLIP) in smolvlm

```
 lerobot-train   --policy.path=lerobot/smolvla_base   --dataset.repo_id=ethanCSL/svla_koch_sorting_n_stacking_side_front_wrist   --batch_size=16   --steps=40000   --output_dir=outputs/train/svla_koch_sorting_n_stacking_side_front_wrist_unfrozen_vision_encoder   --job_name=my_smolvla_training   --policy.device=cuda   --policy.repo_id=ethanCSL/svla_koch_sorting_n_stacking_side_front_wrist_unfrozen_vision_encoder    --wandb.enable=false   --rename_map='{
    "observation.images.front": "observation.images.camera1",
    "observation.images.wrist":   "observation.images.camera2",
    "observation.images.side":  "observation.images.camera3"
  }'   --dataset.video_backend=pyav --policy.freeze_vision_encoder=false --policy.train_expert_only=false
```

Evaluation:

Evaluation with three cameras

```
lerobot-record   --robot.type=koch_follower   --robot.port=/dev/ttyUSB_follower   --robot.id=my_awesome_follower_arm   --robot.cameras='{
    camera1: {type: opencv, index_or_path: 5, width: 640, height: 480, fps: 30, fourcc: MJPG},
    camera2: {type: opencv, index_or_path: 8, width: 640, height: 480, fps: 30, fourcc: MJPG},
    camera3: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30, fourcc: MJPG},
  }'   --dataset.single_task="Put the green cube in the box."   --dataset.repo_id=ethanCSL/eval_Ting_grip_block   --dataset.episode_time_s=500000   --dataset.num_episodes=10   --teleop.type=koch_leader   --teleop.port=/dev/ttyUSB_leader   --teleop.id=my_awesome_leader_arm   --policy.path=/home/bruce/CSL/lerobot_nn/outputs/train/Stanley_grip_block_2color/checkpoints/020000/pretrained_model   --policy.empty_cameras=1 --dataset.reset_time_s=5  
```

Evaluation with two cameras

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

Real-time Attention Visualization

```
python src/lerobot/scripts/lerobot_record_realtime_stanley.py   --robot.type=koch_follower   --robot.port=/dev/ttyUSB_follower   --robot.id=my_awesome_follower_arm   --robot.cameras='{
    camera1: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30},
    camera2: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30}
  }'   --dataset.single_task="Put the red cube in the right box, the green cube in the left box."   --dataset.repo_id=ethanCSL/eval_Ting_grip_block   --dataset.episode_time_s=500000   --dataset.num_episodes=10   --teleop.type=koch_leader   --teleop.port=/dev/ttyUSB_leader   --teleop.id=my_awesome_leader_arm   --policy.path=ethanCSL/svla_koch_sorting_only_wrist   --policy.empty_cameras=1 --dataset.reset_time_s=5  --display_data=True
```

#### PI0.5

Training(RTX 5090 is not enough!!)

```
lerobot-train   --dataset.repo_id=ethanCSL/svla_koch_sorting_n_stacking   --policy.type=pi05   --output_dir=./outputs/train/pi05_koch_sorting_n_stacking   --job_name=pi05_sorting_stacking   --policy.repo_id=ethanCSL/pi05_koch_sorting_n_stacking   --policy.pretrained_path=lerobot/pi05_base   --policy.compile_model=true   --policy.gradient_checkpointing=true   --wandb.enable=true   --policy.dtype=bfloat16   --policy.freeze_vision_encoder=false   --policy.train_expert_only=false   --policy.device=cuda   --batch_size=8   --steps=30000 --dataset.video_backend=pyav

```

# SmolVLA VLA Steering Experiment

Record experimental dataset

High intervention roll out for later experiment

```
 python src/lerobot/scripts/lerobot_record_realtime_attention_weight_stanley_steering_test_debug.py \
  --robot.type=koch_follower \
  --robot.port=/dev/ttyUSB_follower \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras='{
    camera1: {type: opencv, index_or_path: 8, width: 640, height: 480, fps: 30, fourcc: MJPG},
    camera2: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30, fourcc: MJPG},
    camera3: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: MJPG}
  }' \
  --dataset.single_task="Put the red cube in the box." \
  --dataset.repo_id=ethanCSL/eval_high_intervention_rollout_debug_trace_height_10eps \
  --dataset.episode_time_s=500000 \
  --dataset.num_episodes=10 \
  --dataset.reset_time_s=10 \
  --dataset.push_to_hub=false \
  --teleop.type=koch_leader \
  --teleop.port=/dev/ttyUSB_leader \
  --teleop.id=my_awesome_leader_arm \
  --policy.path=ethanCSL/svla_koch_pick_n_place_vla_steering_height_test2 \
  --fixed_action.mode=off \
  --action_trace.enabled=true \
  --intervention.label=high \
  --intervention.name=high_transport \
  --intervention.alpha=6.0 \
  --intervention.enable_steering=true
```

Low intervention roll out for later experiment

```
 python src/lerobot/scripts/lerobot_record_realtime_attention_weight_stanley_steering_test_debug.py \
  --robot.type=koch_follower \
  --robot.port=/dev/ttyUSB_follower \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras='{
    camera1: {type: opencv, index_or_path: 8, width: 640, height: 480, fps: 30, fourcc: MJPG},
    camera2: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30, fourcc: MJPG},
    camera3: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: MJPG}
  }' \
  --dataset.single_task="Put the red cube in the box." \
  --dataset.repo_id=ethanCSL/eval_low_intervention_rollout_debug_trace_height_10eps \
  --dataset.episode_time_s=500000 \
  --dataset.num_episodes=10 \
  --dataset.reset_time_s=10 \
  --dataset.push_to_hub=false \
  --teleop.type=koch_leader \
  --teleop.port=/dev/ttyUSB_leader \
  --teleop.id=my_awesome_leader_arm \
  --policy.path=ethanCSL/svla_koch_pick_n_place_vla_steering_height_test2 \
  --fixed_action.mode=off \
  --action_trace.enabled=true \
  --intervention.label=low \
  --intervention.name=low_transport \
  --intervention.alpha=4.0 \
  --intervention.enable_steering=true
```

No intervention roll out for later experiment

```
 python src/lerobot/scripts/lerobot_record_realtime_attention_weight_stanley_steering_test_debug.py \
  --robot.type=koch_follower \
  --robot.port=/dev/ttyUSB_follower \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras='{
    camera1: {type: opencv, index_or_path: 8, width: 640, height: 480, fps: 30, fourcc: MJPG},
    camera2: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30, fourcc: MJPG},
    camera3: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: MJPG}
  }' \
  --dataset.single_task="Put the red cube in the box." \
  --dataset.repo_id=ethanCSL/eval_high_intervention_rollout_debug_trace_height_10eps \
  --dataset.episode_time_s=500000 \
  --dataset.num_episodes=10 \
  --dataset.reset_time_s=10 \
  --dataset.push_to_hub=false \
  --teleop.type=koch_leader \
  --teleop.port=/dev/ttyUSB_leader \
  --teleop.id=my_awesome_leader_arm \
  --policy.path=ethanCSL/svla_koch_pick_n_place_vla_steering_height_test2 \
  --fixed_action.mode=off \
  --action_trace.enabled=true \
  --intervention.label=high \
  --intervention.name=high_transport \
  --intervention.alpha=0.0 \
  --intervention.enable_steering=false
```

## First Experiment

Prove same predicted action sequence → nearly same real-robot behavior

### Capture one predicted action sequence from the real policy

```
mkdir -p fixed_action_sequences

python src/lerobot/scripts/lerobot_record_realtime_attention_weight_stanley_steering_test_debug.py \
  --robot.type=koch_follower \
  --robot.port=/dev/ttyUSB_follower \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras='{
    camera1: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30, fourcc: MJPG},
    camera2: {type: opencv, index_or_path: 11, width: 640, height: 480, fps: 30, fourcc: MJPG},
    camera3: {type: opencv, index_or_path: 7, width: 640, height: 480, fps: 30, fourcc: MJPG}
  }' \
  --dataset.single_task="Put the red cube in the box." \
  --dataset.repo_id=ethanCSL/eval_fixed_action_capture_prof_high \
  --dataset.episode_time_s=500000 \
  --dataset.num_episodes=1 \
  --dataset.reset_time_s=5 \
  --dataset.push_to_hub=false \
  --teleop.type=koch_leader \
  --teleop.port=/dev/ttyUSB_leader \
  --teleop.id=my_awesome_leader_arm \
  --policy.path="$POLICY" \
  --fixed_action.mode=capture \
  --fixed_action.path=fixed_action_sequences/high_rest0_to_lift2_prof.pt \
  --fixed_action.start_chunk=0 \
  --fixed_action.stop_chunk=2 \
  --fixed_action.max_steps=240 \
  --intervention.label=high \
  --intervention.name=high_transport \
  --intervention.alpha=6.0 \
  --intervention.enable_steering=true
```

> **Note**
> Put the cube at the selected test position.
> 
> Let the policy run from rest → reach → lift.
>
> Press right key to save episode.
> 
> The script will save the exact robot_action_to_send sequence into fixed_action_sequences/high_rest0_to_lift2_prof.pt.

### Replay predicted action sequence from the real policy

```
python src/lerobot/scripts/lerobot_record_realtime_attention_weight_stanley_steering_test_debug.py \
  --robot.type=koch_follower \
  --robot.port=/dev/ttyUSB_follower \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras='{
    camera1: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30, fourcc: MJPG},
    camera2: {type: opencv, index_or_path: 11, width: 640, height: 480, fps: 30, fourcc: MJPG},
    camera3: {type: opencv, index_or_path: 7, width: 640, height: 480, fps: 30, fourcc: MJPG}
  }' \
  --dataset.single_task="Put the red cube in the box." \
  --dataset.repo_id=ethanCSL/eval_fixed_action_replay_prof_high \
  --dataset.episode_time_s=500000 \
  --dataset.num_episodes=3 \
  --dataset.reset_time_s=10 \
  --dataset.push_to_hub=false \
  --teleop.type=koch_leader \
  --teleop.port=/dev/ttyUSB_leader \
  --teleop.id=my_awesome_leader_arm \
  --policy.path="$POLICY" \
  --fixed_action.mode=replay \
  --fixed_action.path=fixed_action_sequences/high_rest0_to_lift2_prof.pt \
  --fixed_action.replay_slowdown=4.0 \
  --intervention.label=high \
  --intervention.name=high_transport \
  --intervention.alpha=6.0 \
  --intervention.enable_steering=true
```

## Second Experiment

Compare actual eef height and predicted eef height from roll outs

This image below is the trajectory segment will be used for comparision:

<img width="2016" height="1152" alt="sequence_exact_observation_montage" src="https://github.com/user-attachments/assets/fb942c30-2186-4e8a-974a-41f267531927" />

### Show result

```
RUN=debug_runs/20260605_115612_Stable
XML=src/lerobot/scripts/follower.xml
OUT=analysis_online_trace_predicted_vs_actual_20260605_115612_Stable_chunk2

python src/lerobot/scripts/make_online_trace_predicted_vs_actual_eef_height.py \
  --run "$RUN" \
  --xml "$XML" \
  --out "$OUT" \
  --pred-source sent_action \
  --actual-source observation_state \
  --chunk-index 2 \
  --delay 4 \
  --actual-window chunk
```
You will see something like

<img width="2369" height="1316" alt="episode_000000_predicted_vs_actual_by_step" src="https://github.com/user-attachments/assets/166a0f3d-fef7-4870-a151-136f0aeb8775" />

<img width="2130" height="1225" alt="paper_predicted_vs_actual_max_eef_height" src="https://github.com/user-attachments/assets/51b6b060-9489-4c94-98fe-5dc9f77c4487" />


# SmolVLA & PI0 & OpenVLA Semantic Top Token Experiment

SmolVLA KNN Method

```
python src/lerobot/scripts/lerobot_reord_top_token.py \
  --policy_family smolvla \
  --policy_path lerobot/smolvla_base \
  --mode all \
  --concepts fast slow high low \
  --plot_prefix smolvla_topk \
  --results_json smolvla_results.json
```

SmolVLA Keyword Method

```
cd ~/CSL/lerobot && conda activate lerobot
python src/lerobot/scripts/lerobot_reord_top_token.py \
  --policy_family smolvla \
  --policy_path lerobot/smolvla_base \
  --mode keyword \
  --keywords_json concept_keywords.json \
  --top_k_tokens 10 \
  --keyword_results_json smolvla_keyword_results.json
```

PI0

```
conda activate lerobot-pi0
cd ~/CSL/lerobot
```

```
python src/lerobot/scripts/lerobot_reord_top_token.py \
  --policy_family pi0 \
  --policy_path lerobot/pi0_base \
  --mode all \
  --concepts fast slow high low \
  --plot_prefix pi0_topk \
  --results_json pi0_results.json
```

PI0 Keyword Method

```
cd ~/CSL/lerobot && conda activate lerobot-pi0
python src/lerobot/scripts/lerobot_reord_top_token.py \
  --policy_family pi0 \
  --policy_path lerobot/pi0_base \
  --mode keyword \
  --keywords_json concept_keywords.json \
  --top_k_tokens 10 \
  --keyword_results_json pi0_keyword_results.json
```


OpnVLA

```
conda activate lerobot-pi0
cd ~/CSL/lerobot
```

```
OMP_NUM_THREADS=4 python src/lerobot/scripts/lerobot_reord_top_token.py \
  --policy_family openvla \
  --policy_path openvla/openvla-7b \
  --mode all \
  --concepts fast slow high low \
  --plot_prefix openvla_topk \
  --results_json openvla_results.json
```

OpenVLA Keyword Method

```
cd ~/CSL/lerobot && conda activate lerobot-pi0
python src/lerobot/scripts/lerobot_reord_top_token.py \
  --policy_family openvla \
  --policy_path openvla/openvla-7b \
  --mode keyword \
  --keywords_json concept_keywords.json \
  --top_k_tokens 10 \
  --keyword_results_json openvla_keyword_results.json
```

Cross-policy Experimental Result

```
python src/lerobot/scripts/lerobot_reord_top_token.py \
  --mode combine_keyword \
  --combine_inputs smolvla_keyword_results.json pi0_keyword_results.json openvla_keyword_results.json \
  --combine_output_prefix vla_comparison
```


PI0-FAST 

Train with LoRA

```
lerobot-pi0fast
cd ~/CSL/lerobot
```

```
lerobot-train   --dataset.repo_id=ethanCSL/svla_koch_pick_n_place_vla_steering_height_test2   --policy.type=pi0_fast   --policy.pretrained_path=/home/csl/CSL/pi0fast-base-fixed   --policy.use_lora=true   --policy.lora_r=64   --policy.lora_alpha=64   --policy.optimizer_lr=2.5e-4   --policy.dtype=bfloat16   --policy.gradient_checkpointing=true   --policy.chunk_size=10   --policy.n_action_steps=10   --policy.max_action_tokens=256   --policy.device=cuda   --batch_size=16   --steps=10000   --output_dir=outputs/train/pi0fast_koch_pick_n_place_vla_steering_height_test2   --job_name=pi0fast_koch_pick_n_place_steering_height_lora_v2   --policy.repo_id=ethanCSL/pi0fast_koch_pick_n_place_vla_steering_height_test2   --wandb.enable=false

```

Evaluate by Server and Client

Server

```
cd ~/CSL/lerobot-pi0fast && conda activate lerobot-pi0fast
python -m lerobot.async_inference.policy_server   --host=0.0.0.0   --port=8080   --fps=30
```

Client

```
cd ~/CSL/lerobot-pi0fast && conda activate lerobot-pi0fast
python -m lerobot.async_inference.robot_client   --robot.type=koch_follower   --robot.port=/dev/ttyUSB_follower   --robot.id=my_awesome_follower_arm   --robot.cameras='{
    front: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30, fourcc: MJPG},
    top: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30, fourcc: MJPG},
    wrist: {type: opencv, index_or_path: 8, width: 640, height: 480, fps: 30, fourcc: MJPG},
  }'   --policy_type=pi0_fast   --pretrained_name_or_path=ethanCSL/pi0fast_koch_pick_n_place_vla_steering_height_test2   --policy_device=cuda   --client_device=cpu   --actions_per_chunk=10   --task="Put the red cube in the box."   --server_address=10.100.4.125:8080   --fps=30
```

# Simulation Benchmark

```
 python src/lerobot/scripts/lerobot_replay_in_mujoco.py \
  --repo_id ethanCSL/svla_koch_pick_n_place_vla_steering_height_test2 \
  --xml follower.xml \
  --episode 1 \
  --stride 1 \
  --width 960 --height 720 \
  --live
```

Automatically generate high and low EEF trajectory episodes

```
python src/lerobot/scripts/collect_libero_height_demos.py     --arc both --n-eps 30 --task-idx 0 --save-video 2>/dev/null
```

Convert HDF5 to LeRobot format

```
python src/lerobot/scripts/convert_hdf5_to_lerobot.py     --hdf5 /home/bruce/datasets/libero_height_demos/libero_spatial/high/task_00.hdf5            /home/bruce/datasets/libero_height_demos/libero_spatial/low/task_00.hdf5     --output ~/datasets/lerobot/libero_height_task00     --task "Pick up the black bowl and place it on the plate."     --cameras agentview     --fps 20     --format libero
```

Training

```
lerobot-train   --policy.path=lerobot/smolvla_base   --dataset.repo_id=ethanCSL/svla_franka_pick_n_place_vla_steering_libero   --batch_size=8   --steps=20000   --output_dir=outputs/train/svla_franka_pick_n_place_vla_steering_libero   --job_name=my_smolvla_training   --policy.device=cuda   --policy.repo_id=ethanCSL/svla_franka_pick_n_place_vla_steering_libero   --wandb.enable=false  --rename_map='{                                              
    "observation.images.agentview": "observation.images.camera1"
  }'   --policy.empty_cameras=2 --dataset.video_backend=pyav

```
Neurons finding

```
python src/lerobot/scripts/libero_find_height_neurons.py \
  --policy-path ethanCSL/svla_franka_pick_n_place_vla_steering_libero \
  --top-n 20 \
  --top-k-tokens 10 \
  --output outputs/libero_height_neurons.json
```

Evaluation

```
python src/lerobot/scripts/libero_eval_steering.py \
  --policy-path ethanCSL/svla_franka_pick_n_place_vla_steering_libero \
  --hdf5 /home/bruce/datasets/libero_height_demos/libero_spatial/high/task_00.hdf5 \
         /home/bruce/datasets/libero_height_demos/libero_spatial/low/task_00.hdf5 \
  --task "Pick up the black bowl and place it on the plate." \
  --conditions none keyword_high keyword_low \
  --neurons-json outputs/libero_height_neurons.json \
  --keyword-alpha 4.0 \
  --n-rollouts 20 \
  --save-video \
  --out-dir outputs/libero_eval_keyword
```
