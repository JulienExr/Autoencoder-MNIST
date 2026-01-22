# Autoencoder (AE)

## Summary and visualizations

The AE is a deterministic encoder/decoder pair trained to minimize reconstruction error (MSE in `training.py`).

- Reconstructions - grid showing original vs reconstructed images.
- Latent PCA -- project the encoded vectors to 2D with PCA and plot colored points by digit label.
- Latent UMAP -- non-linear 2D projection of the latent space (often clearer than PCA).
- Interpolation (2 → 5) -- linear interpolation in latent space between an example "2" and an example "5".
- Sampling from random latent vectors -- decode Gaussian random vectors and inspect outputs.

Example :

<figure style="text-align: center;">
  <img src="../../demo/recon_ae_20.png" alt="AE reconstruction example" style="max-width: 100%;" />
  <figcaption style="font-style: italic;">Figure: Original images (top row) and their reconstructions (bottom row).</figcaption>
</figure>

<figure style="text-align: center;">
  <img src="../../demo/inter_ae_20.png" alt="AE interpolation 2 to 5 example" style="max-width: 100%;" />
  <figcaption style="font-style: italic;">Figure: Decoded images along a linear path in latent space between a sampled "2" and a sampled "5".</figcaption>
</figure>

<figure style="text-align: center;">
  <img src="../../demo/pca_ae_20.png" alt="AE PCA example" style="max-width: 100%;" />
  <figcaption style="font-style: italic;">Figure: PCA 2D projection of AE latent vectors, colored by digit label.</figcaption>
</figure>

- The PCA 2D scatter shows how encoded vectors cluster by digit label. We can observe compact clusters but also empty regions: AE latent space is not forced to follow a known prior, so regions between clusters can be meaningless.

<figure style="text-align: center;">
  <img src="../../demo/umap_ae_20.png" alt="AE UMAP example" style="max-width: 100%;" />
  <figcaption style="font-style: italic;">Figure: UMAP 2D projection of AE latent vectors.</figcaption>
</figure>

- With UMAP, we can see a real separation between digit clusters, but also some curved manifolds and local neighborhoods that PCA may not reveal as clearly.

<figure style="text-align: center;">
  <img src="../../demo/noise_ae_20.png" alt="AE noise sampling example" style="max-width: 100%;" />
  <figcaption style="font-style: italic;">Figure: Random Gaussian latents decoded by the AE.</figcaption>
</figure>

Sampling / noise generation (AE)
- The AE's decoder is trained to reconstruct images from encoded latents produced by its encoder. It is not trained to decode vectors sampled from a standard Gaussian prior. As a result, decoding random Gaussian noise often yields garbage or highly distorted digits.

Why AE sampling fails :
- No regularization: the AE encoder can place encoded points anywhere in latent space; there is no force to match a Gaussian prior.
- Decoder overfits to encoder manifold: decoder learns to map encoder outputs back to images, but random z are off-manifold.
- Conclusion: A plain AE is good for compression and reconstruction, but not a reliable generative model by sampling random latents.

## Fashion-MNIST example renders

Below are example placeholders from the Fashion-MNIST dataset. This dataset is more complex than MNIST, so reconstructions and latent embeddings can look less clean but we still see meaningful structure. 

<figure style="text-align: center;">
  <img src="../../demo/fashion_recon_ae_20.png" alt="Fashion-MNIST AE reconstructions" style="max-width: 100%;" />
  <figcaption style="font-style: italic;">Figure: Fashion-MNIST AE reconstructions after 20 epochs.</figcaption>
</figure>

## Code
- Model: `Encoder`, `Decoder`, `Autoencoder` in `modules/autoencoder/ae.py`.
- Training loop: `modules/autoencoder/training.py`.
- CLI entry point: `main.py`.

## Training
Examples:

- MNIST
  - `python main.py --model AE --dataset mnist --latent_dim 32`
- Fashion-MNIST
  - `python main.py --model AE --dataset fashion_mnist --latent_dim 128`

## Outputs
- Checkpoints: `models/AE/`
- Visuals: `visu/<dataset>_autoencoder/` with `recon/`, `pca/`, `umap/`, `interp/`, `noise/`. ( if enabled in `main.py` )