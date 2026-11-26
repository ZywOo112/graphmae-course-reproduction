"""Degree masking only: keep official corruption, model and loss unchanged."""
import csv
import hashlib
import json
from pathlib import Path
import types

import numpy as np
import torch

ALPHAS = (0.0, 0.25, 0.5, 1.0)

def group_name(alpha):
    return 'control' if alpha == 0 else f'degree_alpha_{alpha:g}'

def array_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def global_rng_hash():
    cpu = array_hash(torch.get_rng_state())
    gpu = array_hash(torch.cuda.get_rng_state()) if torch.cuda.is_available() else None
    return cpu, gpu

def calculate_degrees(edge_index, n_nodes, output):
    """Compute once from structure only, with no labels or split inputs."""
    edges = edge_index.cpu().long()
    without_loops = edges[:, edges[0] != edges[1]]
    keys = without_loops[0] * n_nodes + without_loops[1]
    unique_keys = torch.unique(keys, sorted=True)
    source = unique_keys // n_nodes
    target = unique_keys % n_nodes
    reversed_keys = torch.sort(target * n_nodes + source).values
    symmetric = torch.equal(unique_keys, reversed_keys)
    if not symmetric:
        raise RuntimeError('Graph is not symmetric; review degree definition before training')
    degrees = torch.bincount(source, minlength=n_nodes)
    values, counts = np.unique(degrees.numpy(), return_counts=True)
    cumulative = counts.cumsum()
    # Choose two degree boundaries that minimize imbalance while keeping ties together.
    choices = []
    for i in range(len(values) - 2):
        for j in range(i + 1, len(values) - 1):
            sizes = np.array([cumulative[i], cumulative[j] - cumulative[i], n_nodes - cumulative[j]])
            choices.append((float(((sizes - n_nodes / 3) ** 2).sum()), int(values[i]), int(values[j])))
    _, low_max, medium_max = min(choices)
    groups = np.where(degrees.numpy() <= low_max, 0, np.where(degrees.numpy() <= medium_max, 1, 2)).astype(np.int64)
    output.mkdir()
    np.save(output / 'degrees.npy', degrees.numpy())
    np.save(output / 'groups.npy', groups)
    with (output / 'degree_distribution.csv').open('x', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['degree', 'node_count'])
        writer.writerows(zip(values, counts))
    with (output / 'nodes.csv').open('x', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['node_id', 'degree', 'degree_group'])
        writer.writerows((i, int(d), int(g)) for i, (d, g) in enumerate(zip(degrees.tolist(), groups)))
    metadata = {'node_count': n_nodes, 'input_edge_entries': int(edges.shape[1]),
                'self_loop_entries_removed': int(edges.shape[1] - without_loops.shape[1]),
                'non_self_edge_entries': int(without_loops.shape[1]),
                'unique_directed_neighbor_pairs': len(unique_keys),
                'duplicate_non_self_entries_removed': int(len(keys) - len(unique_keys)),
                'bidirectionally_symmetric': symmetric, 'degree_definition': 'unique non-self neighbors in symmetric graph',
                'minimum': int(degrees.min()), 'maximum': int(degrees.max()),
                'mean': float(degrees.double().mean()), 'median': float(np.median(degrees.numpy())),
                'group_rule': 'minimize sum of squared deviations from N/3 over two whole-degree boundaries; tie: smaller boundaries',
                'low_max_inclusive': low_max, 'medium_max_inclusive': medium_max,
                'groups': [{'group': int(g), 'name': ['low', 'medium', 'high'][g],
                            'nodes': int((groups == g).sum()),
                            'minimum_degree': int(degrees.numpy()[groups == g].min()),
                            'maximum_degree': int(degrees.numpy()[groups == g].max()),
                            'mean_degree': float(degrees.numpy()[groups == g].mean())} for g in range(3)],
                'degree_vector_raw_int64_sha256': array_hash(degrees),
                'files_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()},
                'labels_used': False}
    with (output / 'degree_metadata.json').open('x') as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    return metadata

class DegreeMasking:
    def __init__(self, degrees, degree_alpha, seed):
        assert degree_alpha in ALPHAS
        self.degrees = degrees.cpu().long()
        self.degree_alpha = degree_alpha
        self.seed = seed
        self.generator = torch.Generator(device='cpu').manual_seed(seed)
        self.weights = (self.degrees.double() + 1).pow(degree_alpha)
        self.counts = torch.zeros_like(self.degrees)
        self.calls = 0
        self.last_trace = None

    def install(self, model):
        adapter = self
        def mask_noise(instance, x, mask_rate=0.3):
            return adapter.apply(instance, x, mask_rate)
        model.encoding_mask_noise = types.MethodType(mask_noise, model)

    def apply(self, model, x, mask_rate):
        n_nodes = x.shape[0]
        k = int(mask_rate * n_nodes)
        assert len(self.degrees) == n_nodes
        # Preserve the original first random call on x.device for every arm.
        torch.randperm(n_nodes, device=x.device)
        if self.degree_alpha == 0:
            # Original uniform randperm algorithm; dedicated CPU RNG, not multinomial.
            permutation = torch.randperm(n_nodes, generator=self.generator, device='cpu')
            selected, kept = permutation[:k], permutation[k:]
        else:
            selected = torch.multinomial(self.weights, k, replacement=False, generator=self.generator)
            visible = torch.ones(n_nodes, dtype=torch.bool)
            visible[selected] = False
            kept = visible.nonzero(as_tuple=True)[0]
        assert selected.unique().numel() == k
        self.counts[selected] += 1
        self.calls += 1
        mask_nodes, keep_nodes = selected.to(x.device), kept.to(x.device)
        # The following corruption operations match the locked official source,
        # including its floor rounding (Cora has one selected-but-unchanged node).
        if model._replace_rate > 0:
            num_noise_nodes = int(model._replace_rate * k)
            permutation_mask = torch.randperm(k, device=x.device)
            token_nodes = mask_nodes[permutation_mask[:int(model._mask_token_rate * k)]]
            noise_nodes = mask_nodes[permutation_mask[-int(model._replace_rate * k):]]
            donors = torch.randperm(n_nodes, device=x.device)[:num_noise_nodes]
            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[donors]
        else:
            permutation_mask = torch.empty(0, dtype=torch.long)
            donors = torch.empty(0, dtype=torch.long)
            token_nodes = mask_nodes
            out_x = x.clone()
            out_x[mask_nodes] = 0.0
        out_x[token_nodes] += model.enc_mask_token
        cpu_rng, cuda_rng = global_rng_hash()
        self.last_trace = {'selected_count': k, 'unique_selected_count': int(selected.unique().numel()),
                           'token_count': len(token_nodes), 'donor_count': len(donors),
                           'replacement_permutation_sha256': array_hash(permutation_mask),
                           'replacement_donors_sha256': array_hash(donors),
                           'global_cpu_after_mask': cpu_rng, 'global_cuda_after_mask': cuda_rng,
                           'local_generator_state_sha256': array_hash(self.generator.get_state())}
        return out_x, (mask_nodes, keep_nodes)

    def save_statistics(self, directory, groups):
        with (directory / 'mask_statistics.csv').open('x', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['node_id', 'degree', 'degree_group', 'masked_count', 'mask_rate'])
            writer.writerows((i, int(d), int(g), int(c), int(c) / self.calls) for i, (d, g, c) in enumerate(zip(self.degrees.tolist(), groups, self.counts.tolist())))
        assert int(self.counts.sum()) == self.calls * 1354
        result = []
        counts = self.counts.numpy()
        for group in range(3):
            chosen = np.asarray(groups) == group
            total = int(counts[chosen].sum())
            result.append({'group': group, 'nodes': int(chosen.sum()), 'selections': total,
                           'mean_masked_times_per_node': float(counts[chosen].mean()),
                           'mask_rate_within_group': total / (self.calls * int(chosen.sum())),
                           'share_of_masked_selections': total / int(counts.sum())})
        with (directory / 'mask_group_statistics.json').open('x') as handle:
            json.dump({'epochs': self.calls, 'masked_nodes_per_epoch': 1354, 'groups': result}, handle, indent=2)
            handle.write('\n')
        return result
