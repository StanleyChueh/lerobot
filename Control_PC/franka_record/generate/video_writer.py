# ### video_writer.py
# '''
# video needs to convert to libx264 decoede to visualize with html
# '''
# import cv2
# import tempfile
# import subprocess
# from pathlib import Path
# import numpy as np

# class VideoWriterModule:
#     """
#     Handles saving a sequence of frames (np.ndarray) to an MP4 file with H.264 codec.
#     """
#     def __init__(self, fps: int):
#         self.fps = fps

#     def write_episode(self, frames: np.ndarray, output_path: Path) -> None:
#         """
#         Save frames to output_path as H.264 encoded MP4.
#         frames: np.ndarray of shape (N, H, W, 3), dtype=uint8
#         """
#         if frames.size == 0:
#             return
#         height, width = frames.shape[1], frames.shape[2]

#         with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as tmpfile:
#             avi_path = tmpfile.name

#         # Step 1: Save as .avi using OpenCV
#         fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#         writer = cv2.VideoWriter(avi_path, fourcc, self.fps, (width, height))
#         if not writer.isOpened():
#             raise RuntimeError(f"Cannot open temp avi writer for {avi_path}")
#         for frame in frames:
#             writer.write(frame)
#         writer.release()

#         # Step 2: Use ffmpeg to re-encode as .mp4 (H.264)
#         output_path = Path(output_path)
#         output_path.parent.mkdir(parents=True, exist_ok=True)
#         cmd = [
#             'ffmpeg',
#             '-y',  # Overwrite
#             '-i', avi_path,
#             '-vcodec', 'libx264',
#             '-pix_fmt', 'yuv420p',
#             str(output_path)
#         ]
#         result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         if result.returncode != 0:
#             print(result.stderr.decode())
#             raise RuntimeError("FFmpeg failed to convert video.")
        
#         Path(avi_path).unlink()  # Clean up
import cv2
import tempfile
import subprocess
from pathlib import Path

class VideoWriterModule:
    def __init__(self, fps: int):
        self.fps = fps

    def write_episode(self, frame_paths, output_path: Path) -> None:
        """
        frame_paths: iterable of image file paths (strings or Path). Only one frame loaded at a time.
        Writes to temp .avi with MJPG then re-encodes to mp4 (libx264) with ffmpeg.
        """
        frame_paths = list(frame_paths)
        if len(frame_paths) == 0:
            return

        # read first to infer size
        first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
        if first is None:
            raise RuntimeError(f"Cannot read first frame: {frame_paths[0]}")
        height, width = first.shape[:2]

        with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as tmpfile:
            avi_path = tmpfile.name

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(avi_path, fourcc, self.fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open temp avi writer for {avi_path}")

        for p in frame_paths:
            frame = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if frame is None:
                print(f"Warning: skipping missing frame {p}")
                continue
            # resize if needed
            if frame.shape[0] != height or frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
        writer.release()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            'ffmpeg', '-y', '-i', avi_path,
            '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', str(output_path)
        ]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if r.returncode != 0:
            print(r.stderr.decode(errors='ignore'))
            raise RuntimeError("FFmpeg failed to convert video.")
        Path(avi_path).unlink()
