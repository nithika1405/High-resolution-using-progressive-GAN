import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from gan.generator import Generator
from gan.discriminator import Discriminator
from gan.utils import save_samples

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

Z_DIM = 256
IMAGE_SIZES = [4, 8, 16, 32, 64, 128, 256]
EPOCHS = [2, 2, 2, 2, 2, 2, 2]  # keep small for testing

DATASET_PATH = "dataset"

def get_loader(size):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    dataset = torchvision.datasets.ImageFolder(DATASET_PATH, transform=transform)
    return DataLoader(dataset, batch_size=16, shuffle=True)

gen = Generator(Z_DIM).to(DEVICE)
disc = Discriminator().to(DEVICE)

opt_gen = optim.Adam(gen.parameters(), lr=1e-3, betas=(0.0, 0.99))
opt_disc = optim.Adam(disc.parameters(), lr=1e-3, betas=(0.0, 0.99))

for step, size in enumerate(IMAGE_SIZES):
    loader = get_loader(size)
    alpha = 0

    for epoch in range(EPOCHS[step]):
        for real, _ in loader:
            real = real.to(DEVICE)
            batch = real.shape[0]

            noise = torch.randn(batch, Z_DIM, 1, 1).to(DEVICE)
            fake = gen(noise, step, alpha)

            # Discriminator
            loss_disc = -(torch.mean(disc(real, step, alpha)) -
                          torch.mean(disc(fake.detach(), step, alpha)))

            opt_disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            # Generator
            loss_gen = -torch.mean(disc(fake, step, alpha))

            opt_gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            alpha += batch / (len(loader.dataset) * EPOCHS[step])
            alpha = min(alpha, 1)

        print(f"Step {step} Epoch {epoch} Alpha {alpha:.3f}")

        save_samples(gen, step, alpha, Z_DIM, DEVICE)

print("Training Complete 🚀")