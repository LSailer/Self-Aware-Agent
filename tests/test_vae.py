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

class TestVAE(unittest.TestCase):
    """Tests for the VAE (Variational Autoencoder) model."""
    
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
        
        # Create a simple test image (random noise)
        self.test_image = torch.rand(3, 64, 64).to(self.device)  # (C, H, W)
        self.batch_test_images = torch.rand(5, 3, 64, 64).to(self.device)  # (B, C, H, W)
    
    def test_vae_encode_single_image(self):
        """Test that the VAE can encode a single image."""
        # Add batch dimension to the test image
        test_image_batch = self.test_image.unsqueeze(0)  # (1, C, H, W)
        
        # Encode the image
        mu, logvar = self.vae.encode(test_image_batch)
        
        # Check shapes
        self.assertEqual(mu.shape, (1, self.latent_dim))
        self.assertEqual(logvar.shape, (1, self.latent_dim))
        
        # Check that values are reasonable
        self.assertTrue(torch.all(torch.isfinite(mu)))
        self.assertTrue(torch.all(torch.isfinite(logvar)))
        
        # Reparameterize to get a latent vector
        z = self.vae.reparameterize(mu, logvar)
        self.assertEqual(z.shape, (1, self.latent_dim))
    
    def test_vae_encode_batch(self):
        """Test that the VAE can encode a batch of images."""
        # Encode the batch of images
        mu, logvar = self.vae.encode(self.batch_test_images)
        
        # Check shapes
        batch_size = self.batch_test_images.shape[0]
        self.assertEqual(mu.shape, (batch_size, self.latent_dim))
        self.assertEqual(logvar.shape, (batch_size, self.latent_dim))
        
        # Check that values are reasonable
        self.assertTrue(torch.all(torch.isfinite(mu)))
        self.assertTrue(torch.all(torch.isfinite(logvar)))
        
        # Reparameterize to get latent vectors
        z = self.vae.reparameterize(mu, logvar)
        self.assertEqual(z.shape, (batch_size, self.latent_dim))
    
    def test_vae_decode_single_vector(self):
        """Test that the VAE can decode a single latent vector back to an image."""
        # Create a random latent vector
        z = torch.randn(1, self.latent_dim).to(self.device)
        
        # Decode the latent vector
        decoded_image = self.vae.decode(z)
        
        # Check shape of the decoded image
        self.assertEqual(decoded_image.shape, (1, 3, 64, 64))
        
        # Check values are in the expected range [0, 1] (due to sigmoid activation)
        self.assertTrue(torch.all(decoded_image >= 0))
        self.assertTrue(torch.all(decoded_image <= 1))
    
    def test_vae_decode_batch(self):
        """Test that the VAE can decode a batch of latent vectors."""
        # Create a batch of random latent vectors
        batch_size = 5
        z_batch = torch.randn(batch_size, self.latent_dim).to(self.device)
        
        # Decode the batch of latent vectors
        decoded_images = self.vae.decode(z_batch)
        
        # Check shape of the decoded images
        self.assertEqual(decoded_images.shape, (batch_size, 3, 64, 64))
        
        # Check values are in the expected range [0, 1] (due to sigmoid activation)
        self.assertTrue(torch.all(decoded_images >= 0))
        self.assertTrue(torch.all(decoded_images <= 1))
    
    def test_vae_reconstruction(self):
        """Test the full VAE reconstruction pipeline (encode -> decode)."""
        # Add batch dimension to the test image
        test_image_batch = self.test_image.unsqueeze(0)  # (1, C, H, W)
        
        # Forward pass through the VAE
        reconstructed, original, mu, logvar = self.vae(test_image_batch)
        
        # Check shapes
        self.assertEqual(reconstructed.shape, test_image_batch.shape)
        self.assertEqual(original.shape, test_image_batch.shape)
        self.assertEqual(mu.shape, (1, self.latent_dim))
        self.assertEqual(logvar.shape, (1, self.latent_dim))
        
        # Check reconstruction loss
        # Compute MSE reconstruction loss
        mse_loss = torch.nn.functional.mse_loss(reconstructed, test_image_batch)
        
        # Loss should be finite
        self.assertTrue(torch.isfinite(mse_loss))
        
        # For a completely random initialization, the loss would be high
        # For a pre-trained model, we would expect a lower loss
        # Here we just check it's a valid value, not the specific value
        self.assertTrue(mse_loss >= 0)

if __name__ == '__main__':
    unittest.main() 