# import time
# import mujoco
# import mujoco.viewer
# import numpy as np

# model = mujoco.MjModel.from_xml_path("follower.xml")
# data = mujoco.MjData(model)

# last_qpos = None

# with mujoco.viewer.launch_passive(model, data) as viewer:
#     print("Adjust the robot with sliders.")
#     print("When it matches the real robot, copy the qpos printed in terminal.")
#     while viewer.is_running():
#         mujoco.mj_step(model, data)
#         viewer.sync()

#         qpos = np.array(data.qpos)
#         if last_qpos is None or np.max(np.abs(qpos - last_qpos)) > 1e-4:
#             print("qpos =", qpos.tolist())
#             last_qpos = qpos.copy()

#         time.sleep(0.01)
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("follower.xml")
data = mujoco.MjData(model)

data.qpos[:] = [1.6, 0, 0, 0, 0, 0]
data.ctrl[:] = [1.6, 0, 0, 0, 0, 0]
mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()