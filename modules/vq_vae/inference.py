import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


from vq_vae import build_vqvae
from tranformer_prior import build_transformer_prior

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def sample_codebook_indices(
	transformer,
	vqvae,
	latent_shape,
	num_images=10,
	temperature=1.0,
	top_k=None,
	device="cuda",
):
	"""Sample a sequence of codebook indices with the Transformer prior."""
	transformer.to(device).eval()
	vqvae.to(device).eval()

	if latent_shape is None or len(latent_shape) != 3:
		raise ValueError("latent_shape must be provided as (C, H, W)")

	_, h, w = latent_shape
	latent_hw = h * w

	num_embeddings = vqvae.vector_quantizer.num_embeddings
	vocab_size = transformer.token_embedding.num_embeddings
	max_seq_len = transformer.position_embedding.num_embeddings

	use_sos = vocab_size == (num_embeddings + 1) and max_seq_len == (latent_hw + 1)
	if max_seq_len < latent_hw:
		raise ValueError(
			f"Transformer seq_len ({max_seq_len}) is smaller than latent grid ({latent_hw})."
		)

	if use_sos:
		seq = torch.full((num_images, 1), num_embeddings, device=device, dtype=torch.long)
		total_len = latent_hw + 1
	else:
		seq = torch.randint(0, num_embeddings, (num_images, 1), device=device, dtype=torch.long)
		total_len = latent_hw

	for _ in range(seq.size(1), total_len):
		logits = transformer(seq)
		next_logits = logits[:, -1, :] / max(temperature, 1e-6)
		if top_k is not None and top_k > 0:
			k = min(top_k, next_logits.size(-1))
			values, indices = torch.topk(next_logits, k, dim=-1)
			masked = torch.full_like(next_logits, float('-inf'))
			masked.scatter_(1, indices, values)
			next_logits = masked
		probs = torch.softmax(next_logits, dim=-1)
		next_token = torch.multinomial(probs, num_samples=1)
		seq = torch.cat([seq, next_token], dim=1)

	if use_sos:
		seq = seq[:, 1:]

	return seq[:, :latent_hw]


@torch.no_grad()
def generate_images_with_transformer(
	transformer,
	vqvae,
	latent_shape,
	num_images=10,
	temperature=1.0,
	top_k=None,
	device="cuda",
):
	"""Generate images by sampling Transformer prior then decoding with VQ-VAE."""
	indices = sample_codebook_indices(
		transformer,
		vqvae,
		latent_shape=latent_shape,
		num_images=num_images,
		temperature=temperature,
		top_k=top_k,
		device=device,
	)

	_, h, w = latent_shape
	embeddings = vqvae.vector_quantizer.embedding(indices)
	quantized = embeddings.view(num_images, h, w, -1).permute(0, 3, 1, 2).contiguous()
	images = vqvae.decoder(quantized)
	return images.detach().cpu()


@torch.no_grad()
def generate_images_for_temperatures(
	transformer,
	vqvae,
	latent_shape,
	temperatures=(0.5, 0.8, 1.0, 1.2, 1.5),
	num_images=10,
	top_k=None,
	device="cuda",
	save_dir="visu/transformer_prior/temps",
):
	"""Generate and save image grids for multiple temperatures."""
	save_path = Path(save_dir)
	save_path.mkdir(parents=True, exist_ok=True)

	results = {}
	for temp in temperatures:
		images = generate_images_with_transformer(
			transformer,
			vqvae,
			latent_shape=latent_shape,
			num_images=num_images,
			temperature=float(temp),
			top_k=top_k,
			device=device,
		)

		fig = plt.figure(figsize=(1.5 * num_images, 2))
		for i in range(num_images):
			plt.subplot(1, num_images, i + 1)
			plt.imshow(images[i, 0], cmap="gray")
			plt.axis("off")
		plt.suptitle(f"Transformer prior samples (T={temp})")
		fig.tight_layout()
		fig.savefig(save_path / f"temp_{temp}.png")
		plt.close(fig)

		results[float(temp)] = images

	return results

if __name__ == "__main__":


	num_embeddings = 64
	embedding_dim = 64
	commitment_cost = 0.25

	vqvae = build_vqvae(num_embeddings=num_embeddings, embedding_dim=embedding_dim, commitment_cost=commitment_cost)

	transformer_state = torch.load("models/TransformerPrior/transformer_mnist.pth", map_location=device)
	vocab_size = transformer_state["token_embedding.weight"].shape[0]
	embedding_dim = transformer_state["token_embedding.weight"].shape[1]
	seq_len = transformer_state["position_embedding.weight"].shape[0]

	transformer = build_transformer_prior(
		vocab_size=vocab_size,
		embedding_dim=embedding_dim,
		num_head=4,
		num_layers=6,
		seq_len=seq_len,
	)

	vqvae.encoder.load_state_dict(
		torch.load("models/VQ-VAE/encoder_mnist.pth", map_location=device),
		strict=True,
	)
	vqvae.decoder.load_state_dict(
		torch.load("models/VQ-VAE/decoder_mnist.pth", map_location=device),
		strict=True,
	)
	vq_path = Path("models/VQ-VAE/vq_mnist.pth")
	if vq_path.exists():
		vqvae.vector_quantizer.load_state_dict(
			torch.load(vq_path, map_location=device),
			strict=True,
		)
	else:
		raise FileNotFoundError(
			"Missing VQ codebook weights. Retrain VQ-VAE to save vq_mnist.pth."
		)
	transformer.load_state_dict(transformer_state, strict=True)

	latent_shape = (embedding_dim, 7, 7)

	generate_images_for_temperatures(
		transformer,
		vqvae,
		latent_shape=latent_shape,
		temperatures=(0.5, 0.8, 1.0, 1.2, 1.5),
		num_images=10,
		top_k=128,
		device=device,
		save_dir="visu/transformer_prior/temps_mnist",
	)
