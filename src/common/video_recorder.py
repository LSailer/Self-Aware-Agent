import cv2
import os
import numpy as np
import torch

class VideoRecorder:
    def __init__(self, filename="output.mp4", resolution=(640, 480), fps=20, log_dir="logs"):
        """Initialize the video recorder."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
        self.filename = os.path.join(log_dir, filename)
        self.out = cv2.VideoWriter(self.filename, fourcc, fps, resolution)

    def write_frame(self, frame):
        """Write a frame to the video."""
        if frame.shape[:2] != (480, 640):  # Ensure correct resolution
            raise ValueError(f"Incorrect frame dimensions: {frame.shape}")
        self.out.write(frame)

    def annotate_frame(self, frame, step, curiosity_reward, self_loss):
            """
            Annotate the video frame with step, curiosity reward, and self loss.
            Args:
                frame (np.ndarray or torch.Tensor): Frame to annotate.
                step (int): Simulation step.
                curiosity_reward (float): Curiosity reward value.
                self_loss (float): Self-model loss value.
            Returns:
                np.ndarray: Annotated BGR frame.
            """
            # Tensor -> numpy
            if isinstance(frame, torch.Tensor):
                frame = frame.cpu().numpy()
            # Channels-first -> HWC
            if frame.ndim == 3 and frame.shape[0] == 3:
                frame = frame.transpose(1, 2, 0)
            # Nur hochskalieren, wenn nicht schon uint8
            if frame.dtype != np.uint8:
                frame = np.uint8(frame * 255)
            # RGB -> BGR für OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            font = cv2.FONT_HERSHEY_SIMPLEX
            text_color = (0, 255, 0)
            thickness = 1
            line_type = cv2.LINE_AA

            cv2.putText(frame, f"Step: {step}", (10, 30), font, 0.5, text_color, thickness, line_type)
            cv2.putText(frame, f"Curiosity: {curiosity_reward:.4f}", (10, 70), font, 0.5, text_color, thickness, line_type)
            cv2.putText(frame, f"Self Loss: {self_loss:.4f}", (10, 90), font, 0.5, text_color, thickness, line_type)

            return frame

    
    def close(self):
        """Release the video writer resources."""
        self.out.release()
        print(f"Video saved as '{self.filename}'")