"""Categorical diffusion, masking prior fusion, and iterative sampling."""

from __future__ import annotations

import math

import torch
from torch import nn

from kale_protein.core.data.tasks.inverse_folding.datasets import AA_ALPHABET, DiffusionBatch, IPABatch

from kale_protein.examples.mapdiff_inverse_folding.egnn import EGNNSequenceDenoiser
from kale_protein.examples.mapdiff_inverse_folding.ipa import IPAMaskPrior


class CategoricalTransition(nn.Module):
    """Stationary categorical transition with uniform or empirical marginal noise."""

    def __init__(self, timesteps=100, vocab_size=20, marginal=None):
        super().__init__()
        if marginal is None:
            marginal = torch.full((vocab_size,), 1.0 / vocab_size)
        marginal = torch.as_tensor(marginal, dtype=torch.float32)
        if marginal.shape != (vocab_size,) or (marginal < 0).any() or marginal.sum() <= 0:
            raise ValueError(f"marginal must be a nonnegative vector of length {vocab_size}.")
        marginal = marginal / marginal.sum()
        steps = torch.arange(timesteps + 1, dtype=torch.float32)
        cosine = torch.cos(((steps / timesteps + 0.008) / 1.008) * math.pi / 2).square()
        alpha_bar = cosine / cosine[0]
        alpha_bar[-1] = 0.0
        self.timesteps = timesteps
        self.vocab_size = vocab_size
        self.register_buffer("marginal", marginal)
        self.register_buffer("alpha_bar", alpha_bar.clamp(0.0, 1.0))

    def matrix(self, alpha):
        eye = torch.eye(self.vocab_size, device=alpha.device, dtype=alpha.dtype)
        stationary = self.marginal.to(alpha).expand(self.vocab_size, -1)
        return alpha[..., None, None] * eye + (1.0 - alpha[..., None, None]) * stationary

    def q_sample(self, x0, node_timesteps):
        probabilities = torch.bmm(
            x0[:, None, :], self.matrix(self.alpha_bar[node_timesteps]).to(x0)
        ).squeeze(1)
        token = torch.multinomial(probabilities.clamp_min(1e-8), 1).squeeze(-1)
        return torch.nn.functional.one_hot(token, self.vocab_size).to(x0)

    def initial_noise(self, nodes, device):
        probabilities = self.marginal.to(device).expand(nodes, -1)
        token = torch.multinomial(probabilities, 1).squeeze(-1)
        return torch.nn.functional.one_hot(token, self.vocab_size).float()

    def posterior(self, z_t, predicted_x0, node_t, node_s):
        alpha_t = self.alpha_bar[node_t]
        alpha_s = self.alpha_bar[node_s]
        ratio = (alpha_t / alpha_s.clamp_min(1e-8)).clamp(0.0, 1.0)
        q_s_bar = self.matrix(alpha_s).to(predicted_x0)
        q_s_to_t = self.matrix(ratio).to(predicted_x0)
        q_t_bar = self.matrix(alpha_t).to(predicted_x0)
        observed = z_t.argmax(dim=-1)
        likelihood = q_s_to_t.gather(2, observed[:, None, None].expand(-1, self.vocab_size, 1)).squeeze(-1)
        denominator = q_t_bar.gather(2, observed[:, None, None].expand(-1, self.vocab_size, 1)).squeeze(-1)
        posterior_given_x0 = q_s_bar * likelihood[:, None, :] / denominator[:, :, None].clamp_min(1e-8)
        probabilities = (predicted_x0[:, :, None] * posterior_given_x0).sum(dim=1)
        return probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-8)


class MapDiffDiffusion(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        egnn_layers=4,
        ipa_layers=3,
        ipa_heads=4,
        ipa_head_dim=24,
        ipa_points=4,
        timesteps=100,
        dropout=0.0,
        marginal=None,
        prior_loss_weight=1.0,
        min_mask_ratio=0.35,
        mask_ratio_deviation=0.25,
    ):
        super().__init__()
        self.denoiser = EGNNSequenceDenoiser(hidden_dim, egnn_layers, dropout=dropout)
        self.prior = IPAMaskPrior(hidden_dim, ipa_layers, ipa_heads, ipa_head_dim, ipa_points, dropout)
        self.transition = CategoricalTransition(timesteps, marginal=marginal)
        self.prior_loss_weight = prior_loss_weight
        self.min_mask_ratio = min_mask_ratio
        self.mask_ratio_deviation = mask_ratio_deviation

    def _prior_inputs(self, batch: DiffusionBatch, predicted_x, entropy, graph_timesteps):
        ipa = batch.ipa
        x = ipa.x.new_zeros(ipa.x.shape)
        x_mask = torch.zeros_like(ipa.x_mask)
        for graph_index in range(batch.graph.num_graphs):
            start, end = batch.graph.ptr[graph_index:graph_index + 2].tolist()
            length = end - start
            x[graph_index, :length] = predicted_x[start:end]
            noise = 1.0 - self.transition.alpha_bar[graph_timesteps[graph_index]]
            ratio = self.min_mask_ratio + self.mask_ratio_deviation * torch.sin(noise * math.pi / 2)
            count = min(length, max(1, int(math.ceil(length * float(ratio)))))
            masked = entropy[start:end].topk(count).indices
            x_mask[graph_index, masked] = 1
            x[graph_index, masked] = 0.0
        return IPABatch(x, ipa.atom_pos, ipa.x_pad, x_mask, ipa.label)

    def predict_logits(
        self, batch: DiffusionBatch, noisy_x, graph_timesteps, conditioning=None
    ):
        base_logits = self.denoiser(
            batch.graph,
            noisy_x,
            graph_timesteps.float(),
            conditioning=conditioning,
        )
        base_probs = torch.softmax(base_logits, dim=-1)
        entropy = -(base_probs * torch.log(base_probs.clamp_min(1e-8))).sum(dim=-1)
        prior_input = self._prior_inputs(batch, base_probs, entropy, graph_timesteps)
        prior_padded = self.prior(prior_input.x, prior_input.atom_pos, prior_input.x_mask, prior_input.x_pad)
        prior_logits = prior_padded[prior_input.x_pad]
        base_confidence = 1.0 - entropy / math.log(base_logits.shape[-1])
        prior_probs = torch.softmax(prior_logits, dim=-1)
        prior_entropy = -(prior_probs * torch.log(prior_probs.clamp_min(1e-8))).sum(dim=-1)
        prior_confidence = 1.0 - prior_entropy / math.log(prior_logits.shape[-1])
        weights = torch.softmax(torch.stack([base_confidence, prior_confidence], dim=-1), dim=-1)
        fused_logits = base_logits * weights[:, :1] + prior_logits * weights[:, 1:]
        return fused_logits, base_logits, prior_logits, prior_input

    def forward(self, batch: DiffusionBatch, conditioning=None):
        graph = batch.graph
        graph_timesteps = torch.randint(1, self.transition.timesteps + 1, (graph.num_graphs,), device=graph.x.device)
        node_timesteps = graph_timesteps[graph.batch]
        noisy_x = self.transition.q_sample(graph.x, node_timesteps)
        fused_logits, base_logits, prior_logits, prior_input = self.predict_logits(
            batch, noisy_x, graph_timesteps, conditioning=conditioning
        )
        target = graph.x.argmax(dim=-1)
        base_loss = torch.nn.functional.cross_entropy(base_logits, target)
        prior_mask = prior_input.x_mask[prior_input.x_pad] == 1
        prior_loss = torch.nn.functional.cross_entropy(prior_logits[prior_mask], target[prior_mask])
        loss = base_loss + self.prior_loss_weight * prior_loss
        return {
            "loss": loss,
            "base_loss": base_loss,
            "prior_loss": prior_loss,
            "logits": fused_logits,
            "noisy_x": noisy_x,
            "timesteps": graph_timesteps,
        }
    def prior_pretrain_loss(self, ipa: IPABatch):
        logits = self.prior(ipa.x, ipa.atom_pos, ipa.x_mask, ipa.x_pad)
        selected = (ipa.x_mask > 0) & ipa.x_pad
        if not selected.any():
            raise ValueError("IPA pretraining batch contains no selected mask candidates.")
        return {"loss": torch.nn.functional.cross_entropy(logits[selected], ipa.label[selected]), "logits": logits}

    @staticmethod
    def _decode(tokens, graph):
        sequences = []
        for index in range(graph.num_graphs):
            start, end = graph.ptr[index:index + 2].tolist()
            sequences.append("".join(AA_ALPHABET[token] for token in tokens[start:end].tolist()))
        return sequences

    @torch.no_grad()
    def sample(
        self,
        batch: DiffusionBatch,
        steps=50,
        method="ddim",
        temperature=1.0,
        num_samples=1,
        conditioning=None,
    ):
        if method not in {"ddim", "ddpm"}:
            raise ValueError("sampling method must be 'ddim' or 'ddpm'.")
        if steps < 1:
            raise ValueError("sampling steps must be at least 1.")
        graph = batch.graph
        schedule = torch.linspace(self.transition.timesteps, 0, min(steps, self.transition.timesteps) + 1, device=graph.x.device)
        schedule = torch.unique_consecutive(schedule.round().long())
        all_sequences = []
        all_trajectories = []
        final_logits = None
        for sample_index in range(num_samples):
            z_t = self.transition.initial_noise(graph.num_nodes, graph.x.device).to(graph.x)
            trajectory = [{"timestep": int(schedule[0]), "sequences": self._decode(z_t.argmax(-1), graph)}]
            for current, following in zip(schedule[:-1], schedule[1:]):
                graph_t = torch.full((graph.num_graphs,), int(current), device=graph.x.device, dtype=torch.long)
                logits, _, _, _ = self.predict_logits(
                    batch, z_t, graph_t, conditioning=conditioning
                )
                probabilities_x0 = torch.softmax(logits / max(float(temperature), 1e-4), dim=-1)
                if int(following) == 0:
                    next_tokens = probabilities_x0.argmax(dim=-1)
                elif method == "ddpm":
                    node_t = graph_t[graph.batch]
                    node_s = torch.full_like(node_t, int(following))
                    reverse_probabilities = self.transition.posterior(z_t, probabilities_x0, node_t, node_s)
                    next_tokens = torch.multinomial(reverse_probabilities.clamp_min(1e-8), 1).squeeze(-1)
                else:
                    alpha_s = self.transition.alpha_bar[int(following)]
                    ddim_probabilities = alpha_s * probabilities_x0 + (1.0 - alpha_s) * z_t
                    next_tokens = ddim_probabilities.argmax(dim=-1)
                z_t = torch.nn.functional.one_hot(next_tokens, 20).to(graph.x)
                trajectory.append({"timestep": int(following), "sequences": self._decode(next_tokens, graph)})
                final_logits = logits
            all_sequences.extend(trajectory[-1]["sequences"])
            all_trajectories.append(trajectory)
        return {
            "sequences": all_sequences,
            "token_ids": z_t.argmax(dim=-1),
            "logits": final_logits,
            "trajectory": all_trajectories[0],
            "trajectories": all_trajectories,
            "sampling_method": method,
        }
