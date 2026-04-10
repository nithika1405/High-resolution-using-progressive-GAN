import torch
import torchvision.utils as vutils
import os

def save_samples(gen, step, alpha, z_dim, device):
    os.makedirs("outputs/samples", exist_ok=True)

    gen.eval()
    with torch.no_grad():
        noise = torch.randn(16, z_dim, 1, 1).to(device)
        fake = gen(noise, step, alpha)

        fake = (fake + 1) / 2

        res = 2 ** (step + 2)
        filename = f"outputs/samples/step{step}_{res}x{res}_alpha{alpha:.2f}.png"

        vutils.save_image(fake, filename, nrow=4)

    gen.train()