import pandas as pd
import matplotlib.pyplot as plt

# 讀取 CSV
file = "/home/csl/franka_ws/logs/franka_tau_log.csv"
file = input()
df = pd.read_csv(file)

# 畫 torque
# plt.figure(figsize=(12,6))
# for i in range(7):
#     plt.plot(df["step"], df[f"tau{i}"], label=f"tau{i}")
# plt.legend()
# plt.title("Joint Torques (tau_d_calculated)")
# plt.xlabel("step")
# plt.ylabel("Torque [Nm]")
# plt.grid(True)
# plt.show()

# 畫 positions
# plt.figure(figsize=(12,6))
# for i in range(7):
#     plt.plot(df["step"], df[f"q{i}"], label=f"q{i}")
#     plt.plot(df["step"], df[f"desired_q{i}"], label = f"desired_q{i}")
#     plt.legend()
#     plt.title("Joint Positions")
#     plt.xlabel("step")
#     plt.ylabel("Position [rad]")
#     plt.grid(True)
#     plt.show()

i = 0

plt.figure(figsize=(12,6))
plt.plot(df["step"], df[f"desired_q{i}"], label = f"desired_q{i}")
plt.plot(df["step"], df[f"q{i}"], label=f"q{i}")
plt.grid(True)
plt.show()


plt.figure(figsize=(12,6))
plt.plot(df["step"], df[f"error_q{i}"])
plt.grid(True)
plt.show()