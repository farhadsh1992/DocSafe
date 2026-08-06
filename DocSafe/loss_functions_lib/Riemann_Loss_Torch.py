


import torch
import torch.nn as nn
import torch.nn.functional as F

class Riemann_Loss(nn.Module):
    """
    PyTorch version of the Riemann loss function for GAN models.
    """
    def __init__(self, batch: int = 1, image_size: int = 256):
        super(Riemann_Loss, self).__init__()

        self.batch = batch
        self.image_size = image_size

    def torch_cov(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes covariance matrix of the given tensor.
        Equivalent to the TensorFlow version `tf_cov`.
        """
        mean_x = torch.mean(x, dim=0, keepdim=True)
        med_x = x - mean_x
        cov_xx = torch.matmul(med_x.T, med_x) / x.shape[0]
        return cov_xx

    def distance_riemann(self, AB: tuple) -> torch.Tensor:
        """
        Computes the Riemannian distance between two covariance matrices.
        """
        A, B = AB
        A = self.torch_cov(A * 255)
        B = self.torch_cov(B * 255)
        
        B, A = torch.abs(B + 1e-12), torch.abs(A + 1e-12)
        
        c = A @ torch.linalg.inv(B)  # Equivalent to tf.math.reciprocal(B) in matrix form
        dist = torch.real(torch.trace(torch.log(c) ** 2))
        
        return dist

    def forward(self, original_image: torch.Tensor, generated_image: torch.Tensor, fake_output_disc: torch.Tensor = None) -> torch.Tensor:
        """
        Computes the loss between original and generated images.
        """

        if original_image.shape[1] == 1:  # Check if it's a grayscale image (channel-first format)
            img = original_image.repeat(1, 3, 1, 1)  # Convert grayscale to RGB
            generated = generated_image.repeat(1, 3, 1, 1)
        else:
            img = original_image
            generated = generated_image

        # Reshape from (B, C, H, W) -> (B, H*W, C)
        # img = img.permute(0, 1, 2, 3).reshape(img.shape[0], 3, -1)
        # generated = generated.permute(0, 1, 2, 3).reshape(generated.shape[0], 3, -1)
        img = img.permute(0, 2, 3, 1).reshape(img.shape[0], -1, 3)
        generated = generated.permute(0, 2, 3, 1).reshape(generated.shape[0], -1, 3)

        # Compute distance in Riemannian space
        loss = torch.stack([self.distance_riemann((img[i], generated[i])) for i in range(img.shape[0])])

        # Normalize the loss
        loss_op = loss / (self.batch * 1)

        # Optional additional loss term (similar to binary cross-entropy in TensorFlow)
        if fake_output_disc is not None:
            loss_c = F.binary_cross_entropy_with_logits(torch.ones_like(fake_output_disc), fake_output_disc)
        else:
            loss_c = 0

        # Compute the final loss
        out = torch.mean(loss_op) + loss_c  # Equivalent to keras.ops.mean(loss_op)

        return out
