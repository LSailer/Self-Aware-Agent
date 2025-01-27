import pytest
import numpy as np
import os
import sys

# Füge den Quellcode-Pfad zum Python-Pfad hinzu
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import Environment
@pytest.fixture
def env():
    """Fixture for creating and cleaning up the Environment instance."""
    environment = Environment()
    yield environment
    environment.close()

def test_push_cube(env):
    """Test whether the agent can push the cube, and visualize it in PyBullet."""
    initial_cube_pos = env.get_state()["cube"]["position"]
    env.apply_action("forward")
    for _ in range(50):
        env.step_simulation()
    updated_cube_pos = env.get_state()["cube"]["position"]
    
    # Visual feedback in PyBullet
    print("Initial Cube Position:", initial_cube_pos)
    print("Updated Cube Position:", updated_cube_pos)

    # Ensure the cube position has changed
    assert not np.allclose(initial_cube_pos, updated_cube_pos), "Cube position should change after pushing."

def test_camera_update(env):
    """Test whether the camera updates correctly when the agent moves."""
    initial_camera_view = env.get_camera_image()
    env.apply_action("right")
    for _ in range(50):
        env.step_simulation()
    updated_camera_view = env.get_camera_image()
    
    # Ensure the images are not identical
    assert np.any(initial_camera_view != updated_camera_view), "Camera view should change when the agent moves."
    # Ensure the image dimensions are consistent
    assert initial_camera_view.shape == updated_camera_view.shape, "Camera views should have the same dimensions."



    """Test if the camera updates correctly when the agent moves."""
    initial_camera_view = self.get_camera_image()
    self.apply_action("right")
    for _ in range(50):
        self.step_simulation()
    updated_camera_view = self.get_camera_image()

    # Combine initial and updated views side by side
    combined_view = np.hstack((initial_camera_view, updated_camera_view))

    # Add labels to the combined image
    height, width, _ = combined_view.shape
    label_height = 50

    labeled_image = np.zeros((height + label_height, width, 3), dtype=np.uint8)
    labeled_image[label_height:] = combined_view

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(labeled_image, "Initial Camera View", (10, label_height - 10),
                font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(labeled_image, "Updated Camera View", (width // 2 + 10, label_height - 10),
                font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Display the combined image
    cv2.imshow("Camera Views", labeled_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    def test_push_cube(self):
        """Test if the agent can push the cube."""
        initial_cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
        self.apply_action("forward")
        for _ in range(50):
            self.step_simulation()
        updated_cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
        print(f"Initial Cube Position: {initial_cube_pos}, Updated Cube Position: {updated_cube_pos}")
        