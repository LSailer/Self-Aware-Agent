import cv2
import os
import numpy as np
import torch

class VideoRecorder:
    def __init__(self, filename="output.mp4", resolution=(640, 480), fps=20): # resolution is (width, height)
        """Initialize the video recorder."""
        self.output_filename = filename
        # OpenCV VideoWriter expects (width, height) for resolution
        self.output_resolution = (int(resolution[0]), int(resolution[1])) 
        self.fps = fps
        
        # Ensure the directory for the file exists
        # If filename includes a path, os.path.dirname will extract it.
        # If it's just a filename, os.path.dirname will be empty, so makedirs won't fail.
        file_dir = os.path.dirname(filename)
        if file_dir and not os.path.exists(file_dir): # Only create if dirname is not empty
            os.makedirs(file_dir, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
        try:
            self.out = cv2.VideoWriter(self.output_filename, fourcc, self.fps, self.output_resolution)
            if not self.out.isOpened():
                # This is a critical error, means VideoWriter could not be initialized.
                raise IOError(f"VideoWriter failed to open for {self.output_filename} with resolution {self.output_resolution}. Check codec and permissions.")
        except Exception as e:
            print(f"FATAL: Error initializing VideoWriter for {self.output_filename}: {e}")
            self.out = None # Ensure self.out exists, so calls to write_frame/close don't cause further errors.

    def annotate_frame(self, input_frame_rgb_hwc, step, curiosity_reward, self_loss):
        """
        Annotate the video frame.
        Args:
            input_frame_rgb_hwc (np.ndarray or torch.Tensor): Frame to annotate.
                                     Expected HWC (Height, Width, Channels), RGB. 
                                     If NumPy: uint8 [0-255] or float [0-1].
                                     If Tensor: Can be CHW or HWC, normalized or not.
            step (int): Simulation step.
            curiosity_reward (float): Curiosity reward value.
            self_loss (float): Self-model loss value.
        Returns:
            np.ndarray: Annotated frame (HWC, BGR, uint8), or None if an error occurs.
        """
        if self.out is None:
            # If VideoWriter failed to initialize, we can't process frames.
            # print(f"Video recorder for {self.output_filename} not initialized, skipping annotation.")
            return None

        # Work on a copy to avoid modifying the original observation from the simulation
        processed_frame = input_frame_rgb_hwc.copy()

        # 1. Convert input to NumPy HWC uint8 [0-255] RGB format
        if isinstance(processed_frame, torch.Tensor):
            processed_frame = processed_frame.cpu().detach().numpy()
            if processed_frame.ndim == 3 and processed_frame.shape[0] == 3:  # CHW to HWC
                processed_frame = processed_frame.transpose(1, 2, 0)
            
            # If float (likely normalized [0,1]), scale to [0,255] and convert to uint8
            if processed_frame.dtype in [np.float32, np.float64]:
                processed_frame = np.clip(processed_frame * 255.0, 0, 255).astype(np.uint8)
            elif processed_frame.dtype != np.uint8:
                # print(f"Warning: Tensor input frame has dtype {processed_frame.dtype}. Converting to uint8.")
                processed_frame = processed_frame.astype(np.uint8)
        
        elif isinstance(processed_frame, np.ndarray):
            if processed_frame.dtype in [np.float32, np.float64]: # If float, scale
                processed_frame = np.clip(processed_frame * 255.0, 0, 255).astype(np.uint8)
            elif processed_frame.dtype != np.uint8: # Ensure uint8
                # print(f"Warning: NumPy input frame has dtype {processed_frame.dtype}. Converting to uint8.")
                processed_frame = processed_frame.astype(np.uint8)
            # If already HWC uint8, it's good.
        else:
            print(f"Error: Unsupported frame type for annotation: {type(processed_frame)}")
            return None # Return None if frame type is not recognized

        # Ensure frame is 3-channel (RGB) before color conversion or annotation
        if processed_frame.ndim == 2: # Grayscale
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2RGB)
        elif processed_frame.shape[2] == 4: # RGBA, convert to RGB
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGBA2RGB)
        elif processed_frame.shape[2] != 3:
            print(f"Error: Frame for annotation has {processed_frame.shape[2]} channels, expected 3 (RGB). Cannot annotate.")
            return None
            
        # At this point, processed_frame is HWC, uint8, RGB

        # 2. Convert RGB to BGR for OpenCV drawing and writing
        try:
            frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
        except cv2.error as e:
            print(f"Error converting frame from RGB to BGR: {e}. Frame shape: {processed_frame.shape}, dtype: {processed_frame.dtype}")
            return None # Return None if color conversion fails

        # 3. Annotate the BGR frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = (0, 255, 0)  # Green in BGR
        thickness = 1
        line_type = cv2.LINE_AA

        cv2.putText(frame_bgr, f"Step: {step}", (10, 30), font, 0.5, text_color, thickness, line_type)
        # Using slightly different y-positions to avoid overlap
        cv2.putText(frame_bgr, f"Cur: {curiosity_reward:.3f}", (10, 50), font, 0.5, text_color, thickness, line_type) 
        cv2.putText(frame_bgr, f"SelfL: {self_loss:.3f}", (10, 70), font, 0.5, text_color, thickness, line_type)
        
        return frame_bgr # Return HWC, BGR, uint8

    def write_frame(self, frame_bgr_hwc_uint8):
        """
        Write a HWC (Height, Width, Channels), BGR, uint8 frame to the video.
        Resizes if necessary to match VideoWriter's initialized resolution.
        """
        if self.out is None or not self.out.isOpened():
            # print(f"Video recorder for {self.output_filename} not initialized or not open, skipping write_frame.")
            return

        if frame_bgr_hwc_uint8 is None:
            # print(f"write_frame for {self.output_filename} received None, skipping.")
            return

        expected_width, expected_height = self.output_resolution # (width, height)
        
        # Input frame_bgr_hwc_uint8 shape is (height, width, channels)
        if frame_bgr_hwc_uint8.shape[0] != expected_height or frame_bgr_hwc_uint8.shape[1] != expected_width:
            # print(f"Resizing frame from ({frame_bgr_hwc_uint8.shape[1]}, {frame_bgr_hwc_uint8.shape[0]}) to ({expected_width}, {expected_height}) for video writer.")
            try:
                frame_bgr_hwc_uint8 = cv2.resize(frame_bgr_hwc_uint8, (expected_width, expected_height))
            except cv2.error as e:
                print(f"Error resizing frame in write_frame for {self.output_filename}: {e}. Frame shape: {frame_bgr_hwc_uint8.shape}")
                return # Don't write a potentially corrupted frame
        
        if frame_bgr_hwc_uint8.dtype != np.uint8:
            # print(f"Warning: Frame for writing to {self.output_filename} is {frame_bgr_hwc_uint8.dtype}, not uint8. Converting.")
            frame_bgr_hwc_uint8 = frame_bgr_hwc_uint8.astype(np.uint8)

        try:
            self.out.write(frame_bgr_hwc_uint8)
        except Exception as e:
            print(f"Error writing frame to video {self.output_filename}: {e}")

    def close(self):
        """Release the video writer resources."""
        if self.out is not None and self.out.isOpened():
            self.out.release()
            # Check if file exists and has a reasonable size after releasing
            if os.path.exists(self.output_filename) and os.path.getsize(self.output_filename) > 1024: # Check for > 1KB
                print(f"Video saved as '{self.output_filename}' (Size: {os.path.getsize(self.output_filename) / 1024:.2f} KB)")
            elif os.path.exists(self.output_filename):
                 print(f"Video file '{self.output_filename}' was created but is very small (Size: {os.path.getsize(self.output_filename)} bytes). It might be empty or corrupted.")
            else:
                print(f"Video file '{self.output_filename}' was not created.")

        elif self.out is None: # VideoWriter failed to initialize
            print(f"Video recorder for '{self.output_filename}' was not initialized properly. No video saved.")
        else: # Initialized but not opened (should be caught by isOpened() check earlier)
            print(f"Video recorder for '{self.output_filename}' was initialized but not opened. No video saved.")