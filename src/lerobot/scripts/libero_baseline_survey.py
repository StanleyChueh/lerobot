#!/usr/bin/env python3
"""Survey baseline (unsteered) success rate across all tasks in a LIBERO suite for
the paper checkpoint, so we can pick a HIGH-baseline task where steering stays in an
acceptable success range. Loads the policy once."""
import argparse, os, sys
from pathlib import Path
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import lerobot.scripts.libero_osc_eval as osc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-path", required=True)
    ap.add_argument("--input-format", default="standard")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--n-action-steps", type=int, default=10)
    ap.add_argument("--tasks", nargs="+", type=int, default=None)
    args = ap.parse_args()
    osc.INPUT_FORMAT = args.input_format

    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.utils import get_safe_torch_device
    from lerobot.scripts.libero_eval_steering import clear_steering, enable_determinism
    from libero.libero import benchmark

    enable_determinism()
    device = get_safe_torch_device("cuda" if torch.cuda.is_available() else "cpu")
    policy = SmolVLAPolicy.from_pretrained(args.policy_path)
    policy.config.device = str(device); policy.config.n_action_steps = args.n_action_steps
    policy.eval().to(device); clear_steering(policy)
    pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=args.policy_path)

    suite = benchmark.get_benchmark_dict()[args.suite]()
    n_tasks = len(suite.get_task_bddl_files())
    task_ids = args.tasks if args.tasks else list(range(n_tasks))
    print(f"suite={args.suite}  n_action_steps={args.n_action_steps}  n={args.n}")
    results = []
    for ti in task_ids:
        fname = suite.get_task_bddl_files()[ti]
        lang = suite.get_task(ti).language
        demo = osc.find_demo_file(args.suite, fname)
        if demo is None:
            print(f"task {ti}: NO demo init states -> skip"); continue
        inits = osc.load_demo_init_states(demo)
        env = osc.make_env(osc.BDDL_ROOT / args.suite / fname)
        obs0 = env.reset()
        plate_key = next((k for k in obs0 if "plate" in k.lower() and k.endswith("_pos")
                          and "to_robot" not in k), None)
        g = p = 0
        for ri in range(args.n):
            r = osc.run_rollout(env, inits[ri % len(inits)], policy, pre, post,
                                device, lang, args.max_steps, plate_key)
            g += int(r["grasped"]); p += int(r["on_plate"])
        env.close()
        sr = 100.0 * p / args.n
        results.append((ti, sr, 100.0 * g / args.n, lang))
        print(f"task {ti}: grasp={100*g/args.n:.0f}%  SUCCESS={sr:.0f}%  | {lang}", flush=True)
    results.sort(key=lambda r: -r[1])
    print("\n=== ranked by success ===")
    for ti, sr, gr, lang in results:
        print(f"  task {ti}: SR={sr:.0f}%  grasp={gr:.0f}%  {lang}")


if __name__ == "__main__":
    main()
