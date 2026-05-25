'''
python gripper_calibrate_q6.py \
  --open_json real_pose_calibration_outputs/open_state.json \
  --closed_json real_pose_calibration_outputs/closed_state.json \
  --target_open_q6_deg -60.0 \
  --target_closed_q6_deg 1.8
'''
import argparse
import json
import numpy as np


def load_state(path):
    with open(path, "r") as f:
        data = json.load(f)
    return np.asarray(data["state_vec"], dtype=np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--open_json", required=True)
    parser.add_argument("--closed_json", required=True)

    # MuJoCo joint_6 range in follower.xml is roughly [-140.4 deg, 1.8 deg].
    # Start with these values, then tune if visual open/closed is too extreme.
    parser.add_argument("--target_open_q6_deg", type=float, default=0.0)
    parser.add_argument("--target_closed_q6_deg", type=float, default=-90.0)

    args = parser.parse_args()

    open_state = load_state(args.open_json)
    closed_state = load_state(args.closed_json)

    real_open = float(open_state[5])
    real_closed = float(closed_state[5])

    target_open = float(args.target_open_q6_deg)
    target_closed = float(args.target_closed_q6_deg)

    if abs(real_closed - real_open) < 1e-6:
        raise ValueError(
            "Open and closed gripper states are almost identical. "
            "Capture them again with clearly different gripper positions."
        )

    gripper_a = (target_closed - target_open) / (real_closed - real_open)
    gripper_b = target_open - gripper_a * real_open

    print("\n" + "=" * 80)
    print("GRIPPER OPEN/CLOSED CALIBRATION")
    print("=" * 80)
    print(f"real_open_state[5]   = {real_open:.6f}")
    print(f"real_closed_state[5] = {real_closed:.6f}")
    print(f"target_open_q6_deg   = {target_open:.6f}")
    print(f"target_closed_q6_deg = {target_closed:.6f}")

    print("\nUse these constants in your EEF FK code:\n")
    print(f"GRIPPER_A = {gripper_a:.12f}")
    print(f"GRIPPER_B = {gripper_b:.12f}")

    print("\nFormula:")
    print("q6_deg = GRIPPER_A * raw_gripper_state + GRIPPER_B")

    print("\nRecommended state_to_q_rad override:")
    print("q_deg[5] = GRIPPER_A * raw_deg[5] + GRIPPER_B")
    print("q_deg[5] = np.clip(q_deg[5], -140.4, 1.8)")


if __name__ == "__main__":
    main()
