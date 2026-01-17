import torch
import argparse

from training import train_autoencoder, train_vae
from ae import build_autoencoder
from vae import build_vae
from data import get_fashion_mnist_dataloaders, get_mnist_dataloaders

def parse_args():
    parser = argparse.ArgumentParser(description="Train Autoencoder or Variational Autoencoder")
    parser.add_argument('--model', type=str, choices=['AE', 'VAE'], required=True, help="Model type to train: 'AE' or 'VAE'")
    parser.add_argument('--dataset', type=str, choices=['mnist', 'fashion_mnist'], default='mnist', help="Dataset to use: 'mnist' or 'fashion_mnist'")
    parser.add_argument('--latent_dim', type=int, default=32, help="Latent dimension size")

    return parser.parse_args()

def main_AE(dataset='mnist', latent_dim=256):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if dataset == 'mnist':
        train_loader, test_loader = get_mnist_dataloaders(batch_size=128)
    elif dataset == 'fashion_mnist':
        train_loader, test_loader = get_fashion_mnist_dataloaders(batch_size=128)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    autoencoder = build_autoencoder(latent_dim=latent_dim)

    print("Starting training...")
    train_autoencoder(autoencoder, train_loader, test_loader, num_epochs=20, learning_rate=1e-3,
                       device=device, visu_dir=f"{dataset}_autoencoder")

    torch.save(autoencoder.encoder.state_dict(), 'model/AE/encoder.pth')
    torch.save(autoencoder.decoder.state_dict(), 'model/AE/decoder.pth')
    print("Model saved as 'model/AE/encoder.pth' and 'model/AE/decoder.pth'.")

def main_VAE(dataset='mnist', latent_dim=32):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if dataset == 'mnist':
        train_loader, test_loader = get_mnist_dataloaders(batch_size=128)
    elif dataset == 'fashion_mnist':
        train_loader, test_loader = get_fashion_mnist_dataloaders(batch_size=128)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    vae = build_vae(latent_dim=latent_dim, mode="pp")

    print("Start training")
    train_vae(vae, train_loader, test_loader, num_epochs=50, learning_rate=1e-3, latent_dim=latent_dim,
               device=device, visu_dir=f"{dataset}_vae")
    torch.save(vae.encoder.state_dict(), 'model/VAE/encoder.pth')
    torch.save(vae.decoder.state_dict(), 'model/VAE/decoder.pth')
    print("Model saved as 'model/VAE/encoder.pth' and 'model/VAE/decoder.pth'.")

def main():
    args = parse_args()
    if args.model == 'AE':
        main_AE(dataset=args.dataset, latent_dim=args.latent_dim)
    elif args.model == 'VAE':
        main_VAE(dataset=args.dataset, latent_dim=args.latent_dim)

if __name__ == "__main__":
    # By default the script runs AE training first, then VAE training.
    # If you want to run only one, comment the other line.
    # You can choose the dataset between 'mnist' and 'fashion_mnist'
    main()