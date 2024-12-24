import cv2
import os
import numpy as np

class VideoRecorder:
    def __init__(self, filename="output.mp4", resolution=(640, 480), fps=20):
        """Initialize the video recorder."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
        self.out = cv2.VideoWriter(filename, fourcc, fps, resolution)
        self.filename = filename

    def write_frame(self, frame):
        """Write a frame to the video."""
        if frame.shape[:2] != (480, 640):  # Ensure correct resolution
            raise ValueError(f"Incorrect frame dimensions: {frame.shape}")
        self.out.write(frame)

    def annotate_frame(self, frame, step, position, reward, loss):
        """Annotate the frame with key metrics."""
        if frame.dtype != np.uint8:
            frame =  (frame * 255).astype(np.uint8)  # Normalize and convert to uint8 if needed

        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = (255, 255, 255)  # White
        thickness = 1
        line_type = cv2.LINE_AA

        cv2.putText(frame, f"Step: {step}", (10, 30), font, 0.5, text_color, thickness, line_type)
        cv2.putText(frame, f"Reward: {reward:.4f}", (10, 70), font, 0.5, text_color, thickness, line_type)
        cv2.putText(frame, f"Loss: {loss:.4f}", (10, 90), font, 0.5, text_color, thickness, line_type)

        return frame
    
    def close(self):
        """Release the video writer resources."""
        self.out.release()
        print(f"Video saved as '{self.filename}'")