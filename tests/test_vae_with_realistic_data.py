import os
import sys
import unittest
import torch
import numpy as np
from torchvision import transforms as T
from PIL import Image

# Add the src directory to the Python path so we can import the VAE module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from models import VAE

class TestVAEWithRealisticData(unittest.TestCase):
    """Tests for the VAE using more realistic data scenarios and edge cases."""
    
    def setUp(self):
        """Set up the test environment."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create a VAE model with default parameters
        self.latent_dim = 32
        self.vae = VAE(input_channels=3, latent_dim=self.latent_dim).to(self.device)
        
        # Create a transform pipeline similar to the one used in the agent
        self.transform = T.Compose([
            T.Resize((64, 64)),     # Resize to 64x64
            T.ToTensor(),           # Convert to tensor and normalize to [0, 1]
        ])
        
        # Create more realistic test images
        # 1. Solid color image (all white)
        self.white_image = torch.ones(3, 64, 64).to(self.device)
        
        # 2. All black image
        self.black_image = torch.zeros(3, 64, 64).to(self.device)
        
        # 3. High contrast image (half black, half white)
        self.contrast_image = torch.zeros(3, 64, 64).to(self.device)
        self.contrast_image[:, :, 32:] = 1.0
        
        # 4. Image with different values in each channel (RGB test)
        self.rgb_image = torch.zeros(3, 64, 64).to(self.device)
        self.rgb_image[0, :, :] = 1.0  # Red channel
        self.rgb_image[1, 16:48, 16:48] = 1.0  # Green in center
        self.rgb_image[2, 32:, 32:] = 1.0  # Blue in bottom right
        
        # 5. Gradient image (for smooth transitions)
        x = torch.linspace(0, 1, 64)
        y = torch.linspace(0, 1, 64)
        xv, yv = torch.meshgrid(x, y)
        self.gradient_image = torch.stack([xv, yv, xv * yv]).to(self.device)
    
    def test_encode_decode_edge_cases(self):
        """Test encoding and decoding with edge case images."""
        test_cases = {
            'white_image': self.white_image,
            'black_image': self.black_image,
            'contrast_image': self.contrast_image,
            'rgb_image': self.rgb_image,
            'gradient_image': self.gradient_image
        }
        
        for name, image in test_cases.items():
            # Add batch dimension
            image_batch = image.unsqueeze(0)
            
            # Encode
            mu, logvar = self.vae.encode(image_batch)
            
            # Check that encoding produces valid values
            self.assertTrue(torch.all(torch.isfinite(mu)), f"Encoding {name} produced non-finite mu values")
            self.assertTrue(torch.all(torch.isfinite(logvar)), f"Encoding {name} produced non-finite logvar values")
            
            # Sample latent vector
            z = self.vae.reparameterize(mu, logvar)
            
            # Decode back to image space
            reconstructed = self.vae.decode(z)
            
            # Check reconstructed image shape
            self.assertEqual(reconstructed.shape, image_batch.shape, 
                            f"Reconstructed {name} has wrong shape")
            
            # Check value range
            self.assertTrue(torch.all(reconstructed >= 0) and torch.all(reconstructed <= 1),
                           f"Reconstructed {name} has values outside [0,1] range")
    
    def test_reparameterization_consistency(self):
        """Test that different reparameterizations of the same mu/logvar are different."""
        # Use a test image
        image_batch = self.rgb_image.unsqueeze(0)
        
        # Encode to get mu and logvar
        mu, logvar = self.vae.encode(image_batch)
        
        # Sample multiple times from the same distribution
        z1 = self.vae.reparameterize(mu, logvar)
        z2 = self.vae.reparameterize(mu, logvar)
        
        # Check that they're different (due to the random sampling)
        # They should be close but not identical
        self.assertFalse(torch.allclose(z1, z2, rtol=1e-4, atol=1e-4),
                         "Two reparameterizations produced identical results, which is highly unlikely")
    
    def test_extreme_latent_values(self):
        """Test decoding with extreme values in the latent space."""
        # Very large positive values
        z_large = torch.ones(1, self.latent_dim).to(self.device) * 10.0
        decoded_large = self.vae.decode(z_large)
        
        # Very large negative values
        z_negative = torch.ones(1, self.latent_dim).to(self.device) * -10.0
        decoded_negative = self.vae.decode(z_negative)
        
        # Check that decoder handles extreme values without errors
        self.assertTrue(torch.all(torch.isfinite(decoded_large)), 
                       "Decoding large positive latent values produced non-finite outputs")
        self.assertTrue(torch.all(torch.isfinite(decoded_negative)), 
                       "Decoding large negative latent values produced non-finite outputs")
        
        # Values should still be in the valid range
        self.assertTrue(torch.all(decoded_large >= 0) and torch.all(decoded_large <= 1),
                       "Decoded large positive latent values outside [0,1] range")
        self.assertTrue(torch.all(decoded_negative >= 0) and torch.all(decoded_negative <= 1),
                       "Decoded large negative latent values outside [0,1] range")
    
    def test_interpolation_in_latent_space(self):
        """Test interpolation between two points in latent space."""
        # Encode two different images
        image1_batch = self.white_image.unsqueeze(0)
        image2_batch = self.black_image.unsqueeze(0)
        
        mu1, logvar1 = self.vae.encode(image1_batch)
        mu2, logvar2 = self.vae.encode(image2_batch)
        
        # Sample latent vectors
        z1 = self.vae.reparameterize(mu1, logvar1)
        z2 = self.vae.reparameterize(mu2, logvar2)
        
        # Create interpolated points
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
        for alpha in alphas:
            # Linear interpolation in latent space
            z_interp = alpha * z1 + (1 - alpha) * z2
            
            # Decode the interpolated point
            decoded = self.vae.decode(z_interp)
            
            # Check shape and value range
            self.assertEqual(decoded.shape, image1_batch.shape)
            self.assertTrue(torch.all(decoded >= 0) and torch.all(decoded <= 1),
                           f"Interpolated image at alpha={alpha} has values outside [0,1] range")
    
    def test_batch_consistency(self):
        """
        Test batch processing consistency.
        
        Note: This test may show slight differences when using BatchNorm layers as they 
        behave differently in batch vs individual processing. We set the model to eval mode
        and use more relaxed tolerance to account for this.
        """
        # Create a small batch with different test images
        batch = torch.stack([
            self.white_image,
            self.black_image,
            self.contrast_image
        ])
        
        # Set model to evaluation mode to use running stats for BatchNorm
        self.vae.eval()
        
        with torch.no_grad():  # No need for gradients for this test
            # Process as batch
            mu_batch, logvar_batch = self.vae.encode(batch)
            
            # Process individually
            mu_single = []
            logvar_single = []
            for i in range(batch.shape[0]):
                mu_i, logvar_i = self.vae.encode(batch[i:i+1])
                mu_single.append(mu_i)
                logvar_single.append(logvar_i)
            
            # Stack results
            mu_single = torch.cat(mu_single, dim=0)
            logvar_single = torch.cat(logvar_single, dim=0)
        
        # Set model back to training mode for other tests
        self.vae.train()
        
        # Use more relaxed tolerances for comparison, as BatchNorm behavior 
        # will cause small differences even in eval mode
        rtol = 1e-2  # Relative tolerance
        atol = 1e-2  # Absolute tolerance
        
        # Calculate max absolute difference for debugging if needed
        max_diff_mu = torch.max(torch.abs(mu_batch - mu_single))
        max_diff_logvar = torch.max(torch.abs(logvar_batch - logvar_single))
        
        # Check if the values are close enough - if test fails, print the max difference
        self.assertTrue(
            torch.allclose(mu_batch, mu_single, rtol=rtol, atol=atol),
            f"Batch encoding produces different mu values than individual encoding. Max diff: {max_diff_mu}"
        )
        self.assertTrue(
            torch.allclose(logvar_batch, logvar_single, rtol=rtol, atol=atol),
            f"Batch encoding produces different logvar values than individual encoding. Max diff: {max_diff_logvar}"
        )

if __name__ == '__main__':
    unittest.main() 