import socket
import os
import struct
import time
import torch
import numpy as np
import cv2
import argparse
import json
# from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.act.modeling_act import ACTPolicy
# 0.0.0.0 -> all interfaces

class SmolVLAInferenceServer:
    def __init__(self, ckpt_path, host='0.0.0.0', port=5001, device='cuda', eval_freq=None, task=""):
        self.eval_freq = eval_freq  # Hz
        self.eval_interval = 1.0 / eval_freq if eval_freq and eval_freq > 0 else 0.0

        # Prompt
        self.task = task

        # Device
        self.device = device
        if device == 'cuda' and not torch.cuda.is_available():
            print("⚠️ CUDA 不可用，自動切換為 CPU")
            self.device = 'cpu'
        print("Using:", self.device)

        # Model
        ckpt_path = os.path.abspath(ckpt_path)
        print(f"📂 使用的 ckpt_path: {ckpt_path}")
        self.policy = SmolVLAPolicy.from_pretrained(ckpt_path, local_files_only=True)
        self.policy.to(self.device)
        self.policy.eval()

        # 小幅度加速（影像尺寸固定時有效）
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

        # Server socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 建議打開重用位址與較大的 backlog（視需要）
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((host, port))
        self.server_socket.listen(8)
        print(" SmolVLA 推論伺服器已啟動，等待 client 連線...")
        self.chunk_acount = 0

        # 初次 accept
        self._accept_client()

    # ---- socket helpers ----
    def _apply_conn_opts(self, sock):
        # 在連線 socket 設 TCP_NODELAY 與較大 buffer
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4*1024*1024)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4*1024*1024)
        # （可選）keepalive
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        except Exception:
            pass

    def _accept_client(self):
        self.conn, self.addr = self.server_socket.accept()
        self._apply_conn_opts(self.conn)
        print(" client 已連線:", self.addr)

    def recv_exact(self, n):
        buf = b''
        while len(buf) < n:
            chunk = self.conn.recv(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    def recv_one_message(self):
        t0 = time.time()
        raw_len = self.recv_exact(4)
        if raw_len is None:
            return None, None, 0, 0.0
        data_len = struct.unpack('>I', raw_len)[0]

        raw_type = self.recv_exact(10)
        if raw_type is None:
            return None, None, 0, 0.0
        try:
            data_type = raw_type.decode('utf-8').strip()
        except UnicodeDecodeError:
            raise RuntimeError("Framing desync: invalid type header")

        data = self.recv_exact(data_len)
        if data is None:
            return None, None, 0, 0.0

        return data_type, data, data_len, (time.time() - t0)

    def _decode_image(self, data_bytes, name):
        arr = np.frombuffer(data_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to decode {name}")
        return img

    def send_data(self, data_type, data_bytes):
        hdr_type = data_type.ljust(10).encode('utf-8')
        data_len = struct.pack('>I', len(data_bytes))
        self.conn.sendall(data_len + hdr_type + data_bytes)

    # ---- main loop ----
    # ---- main loop ----
    def run(self):
        last_time = time.time()

        # 初始化 next_tick
        if self.eval_interval > 0:
            next_tick = time.time()
        else:
            next_tick = None

        while True:
            try:
                # ⚠️ 這一行必須在「sleep 完之後」才執行
                loop_t0 = time.time()
                proc_t0 = loop_t0

                # --------------------
                #  1) 接收 state
                # --------------------
                t, d, s1, dt1 = self.recv_one_message()
                if t != 'list':
                    raise RuntimeError(f"Unexpected type for state: {t}")

                state = json.loads(d.decode('utf-8'))
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device, non_blocking=True)

                # --------------------
                #  2) 接收 img1
                # --------------------
                t, d, s2, dt2 = self.recv_one_message()
                if t != 'img1':
                    raise RuntimeError(f"Unexpected type for img1: {t}")

                img1 = self._decode_image(d, "img1")
                img1 = cv2.resize(img1, (640,480))

                # 3) 忽略 img2
                t, d, s3, dt3 = self.recv_one_message()
                if t != 'img2':
                    raise RuntimeError(f"Unexpected type for img2: {t}")

                img2 = self._decode_image(d, "img2")
                img2 = cv2.resize(img2, (640,480))

                # --------------------
                #  4) 接收 img3
                # --------------------
                t, d, s4, dt4 = self.recv_one_message()
                if t != 'img3':
                    raise RuntimeError(f"Unexpected type for img3: {t}")

                img3 = self._decode_image(d, "img3")
                img3 = cv2.resize(img3, (640,480))

                # --------------------
                #  5) 準備 obs
                # --------------------
                obs = {
                    'observation.state': state_tensor,
                    'observation.images.image_front_eye':
                        torch.from_numpy(img1).to(self.device, non_blocking=True)
                            .permute(2, 0, 1).unsqueeze(0).float() / 255.0,
                    'observation.images.image_rear_eye':
                        torch.from_numpy(img2).to(self.device, non_blocking=True)
                            .permute(2, 0, 1).unsqueeze(0).float() / 255.0,
                    'observation.images.image_front_view':
                        torch.from_numpy(img3).to(self.device, non_blocking=True)
                            .permute(2, 0, 1).unsqueeze(0).float() / 255.0,
                    'task': [self.task],
                }

                # --------------------
                #  6) 推論
                # --------------------
                inf_t0 = time.time()
                with torch.no_grad():
                    if self.device == 'cuda':
                        with torch.cuda.amp.autocast():
                            action = self.policy.select_action(obs)
                            self.chunk_acount += 1
                            print(f"self.chunk_acount: {self.chunk_acount}")

                            if self.chunk_acount % 50 == 0:
                                print("finish 50 chunks")
                                self.chunk_acount = 0
                            #     self.policy.reset()
                            #     self.chunk_acount = 0
                            #     print("重置 Transformer 快取")
                    else:
                        action = self.policy.select_action(obs)
                inf_t1 = time.time()

                action_np = action.squeeze(0).detach().cpu().numpy().tolist()

                # --------------------
                #  7) 計算統計
                # --------------------
                now = time.time()
                total_interval = now - last_time
                last_time = now

                total_freq = 1.0 / total_interval if total_interval > 0 else 0.0
                infer_freq = 1.0 / (inf_t1 - inf_t0) if inf_t1 > inf_t0 else 0.0
                proc_interval = inf_t0 - proc_t0
                total_bytes = s1 + s2 + s3 + s4
                total_xfer_time = dt1 + dt2 + dt3 + dt4
                xfer_mbps = (total_bytes / total_xfer_time) / (1024 * 1024) if total_xfer_time > 0 else 0.0

                print(
                    f"推論完成: {np.round(action_np[7],6)} | "
                    f"總頻率: {total_freq:.2f} Hz | "
                    f"純推論頻率: {infer_freq:.2f} Hz | "
                    f"資料處理耗時: {proc_interval:.4f} s | "
                    f"傳輸速率: {xfer_mbps:.2f} MB/s"
                )

                # 8) send back
                self.send_data('list', json.dumps(action_np).encode('utf-8'))

                # ------------------------------
                # 9) 固定頻率控制（一定要在最後）
                # ------------------------------
                if self.eval_interval > 0:
                    next_tick += self.eval_interval
                    sleep_time = next_tick - time.time()
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    else:
                        next_tick = time.time()

            except Exception as e:
                print("❌ 連線/封包錯誤，關閉連線等待重連：", e)
                try:
                    self.conn.close()
                except Exception:
                    pass
                self._accept_client()
                last_time = time.time()
                if self.eval_interval > 0:
                    next_tick = time.time()
                continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', type=str, required=True)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5001)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--eval-freq', type=float, default=0.0)  # 0 = 不節流，追求最高頻率
    parser.add_argument('--task', type=str, default="pick up the red cube and place it",
                    help="Language task/instruction for SmolVLA")
    args = parser.parse_args()

    server = SmolVLAInferenceServer(
        ckpt_path=args.ckpt_path,
        host=args.host,
        port=args.port,
        device=args.device,
        eval_freq=args.eval_freq,
        task=args.task,
    )
    server.run()
