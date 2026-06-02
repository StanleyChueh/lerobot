import argparse
import copy
import torch
from pathlib import Path

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def activation_steering_config():
    # Same high_transport setting as your evaluation script.
    return {
        0: [1293],
        1: [1050],
        3: [2259],
        4: [1183],
        7: [295],
        11: [1115, 1595],
        13: [431],
        14: [736, 805],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_path", required=True)
    parser.add_argument("--chunk_pt", required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--disable_steering", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    policy = SmolVLAPolicy.from_pretrained(args.policy_path)
    policy.to(device)
    policy.eval()

    if not args.disable_steering:
        policy.set_activation_steering(
            steering_neurons=activation_steering_config(),
            alpha=args.alpha,
            record_debug=False,
            enable_steering=True,
        )
    else:
        if hasattr(policy, "clear_activation_steering"):
            policy.clear_activation_steering()

    data = torch.load(args.chunk_pt, map_location="cpu")

    saved_actions = data["actions"].to(device)
    noise = data["noise"].to(device)
    base_batch = move_batch_to_device(data["policy_batch"], device)

    outputs = []

    with torch.no_grad():
        for i in range(args.repeats):
            # Avoid accidental mutation by _get_action_chunk().
            batch_i = {
                k: v.clone() if torch.is_tensor(v) else copy.deepcopy(v)
                for k, v in base_batch.items()
            }

            policy.reset()

            if not args.disable_steering:
                policy.set_activation_steering(
                    steering_neurons=activation_steering_config(),
                    alpha=args.alpha,
                    record_debug=False,
                    enable_steering=True,
                )

            actions_i = policy._get_action_chunk(batch_i, noise.clone())
            outputs.append(actions_i.detach().cpu())

            diff_saved = (actions_i - saved_actions).abs()
            print(
                f"[repeat {i:02d}] vs saved online action: "
                f"max_abs={diff_saved.max().item():.8f}, "
                f"mean_abs={diff_saved.mean().item():.8f}"
            )

    ref = outputs[0]
    print("\n=== Repeated offline output comparison ===")
    for i, out in enumerate(outputs[1:], start=1):
        d = (out - ref).abs()
        print(
            f"repeat {i:02d} vs repeat 00: "
            f"max_abs={d.max().item():.10f}, "
            f"mean_abs={d.mean().item():.10f}, "
            f"allclose_1e-6={torch.allclose(out, ref, atol=1e-6, rtol=1e-6)}"
        )


if __name__ == "__main__":
    main()
