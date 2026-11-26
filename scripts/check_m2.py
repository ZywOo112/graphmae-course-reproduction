"""Verify actual GPU masking, corruption and dropout RNG before any pilot."""
import argparse
import json
import os
import sys

from run_m2 import ROOT, REPO, setup_gpu, load_args, ALPHAS, save_json, set_seed, state_hash

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', required=True)
    cli = parser.parse_args()
    from pathlib import Path
    batch = Path(cli.batch).resolve()
    _, gpu, _, _ = setup_gpu(batch)
    import numpy as np
    import torch
    import torch.nn.functional as functional
    from m2_masking import DegreeMasking, global_rng_hash
    sys.path.insert(0, str(REPO))
    from graphmae.models import build_model
    from graphmae.datasets.data_util import load_dataset
    torch.set_num_threads(4)
    os.chdir(ROOT / 'data')
    graph, _ = load_dataset('cora')
    del graph.y
    del graph.test_mask
    graph = graph.to('cuda:0')
    degrees = torch.from_numpy(np.load(batch / 'degree/degrees.npy'))
    checks = []
    for seed in range(5):
        arms = []
        for alpha in ALPHAS:
            set_seed(seed)
            model = build_model(load_args(seed)).to('cuda:0')
            initial_hash = state_hash(model)
            original_method = type(model).encoding_mask_noise
            sampler = DegreeMasking(degrees, alpha, seed)
            sampler.install(model)
            assert state_hash(model) == initial_hash
            cpu_before = torch.get_rng_state().clone()
            cuda_before = torch.cuda.get_rng_state().clone()
            with torch.no_grad():
                changed, (masked, kept) = model.encoding_mask_noise(graph.x, 0.5)
            assert len(masked) == masked.unique().numel() == 1354
            assert len(kept) == 1354 and not torch.isin(masked, kept).any()
            assert torch.equal(changed[kept], graph.x[kept])
            after_adapter = global_rng_hash()
            assert torch.equal(torch.get_rng_state(), cpu_before), 'Local sampling unexpectedly consumed global CPU RNG'
            # Replay the actual official method with the selected node permutation.
            torch.set_rng_state(cpu_before)
            torch.cuda.set_rng_state(cuda_before)
            original_randperm = torch.randperm
            calls = []
            def replay_randperm(n, *args, **kwargs):
                value = original_randperm(n, *args, **kwargs)
                calls.append({'n': n, 'device': str(kwargs.get('device'))})
                return torch.cat([masked, kept]) if len(calls) == 1 else value
            torch.randperm = replay_randperm
            try:
                with torch.no_grad():
                    official_changed, official_indices = original_method(model, graph.x, 0.5)
            finally:
                torch.randperm = original_randperm
            assert torch.equal(changed, official_changed)
            assert global_rng_hash() == after_adapter
            assert [call['n'] for call in calls] == [2708, 1354, 2708]
            assert sampler.last_trace['token_count'] == 1286 and sampler.last_trace['donor_count'] == 67
            forward_traces = []
            for step in range(3):
                dropout_trace = []
                original_dropout = functional.dropout
                def observed_dropout(value, p=0.5, training=True, inplace=False):
                    before = global_rng_hash()
                    output = original_dropout(value, p=p, training=training, inplace=inplace)
                    dropout_trace.append({'shape': list(value.shape), 'p': p, 'training': training, 'before': before, 'after': global_rng_hash()})
                    return output
                functional.dropout = observed_dropout
                try:
                    model.train()
                    model.zero_grad()
                    loss, _ = model(graph.x, graph.edge_index)
                    assert torch.isfinite(loss)
                    loss.backward()
                    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                finally:
                    functional.dropout = original_dropout
                assert len(dropout_trace) == 6
                forward_traces.append({'step': step, 'dropout': dropout_trace, 'global_after_backward': global_rng_hash(),
                                       'mask_rng': {key: value for key, value in sampler.last_trace.items() if key != 'local_generator_state_sha256'}})
            arms.append({'degree_alpha': alpha, 'initial_model_sha256': initial_hash, 'forward_trace': forward_traces,
                         'official_corruption_exact_for_given_indices': True, 'randperm_calls': calls})
        assert len({record['initial_model_sha256'] for record in arms}) == 1
        assert all(record['forward_trace'] == arms[0]['forward_trace'] for record in arms[1:])
        checks.append({'seed': seed, 'initial_model_sha256': arms[0]['initial_model_sha256'], 'all_four_initial_weights_identical': True,
                       'all_four_global_rng_replacement_dropout_traces_identical': True, 'arms': arms})
    save_json(batch / 'masking_checks.json', {'status': 'passed', 'gpu_uuid': gpu['uuid'], 'actual_cora_graph': True,
                                             'three_forward_backward_steps_per_arm_per_seed': True,
                                             'original_corruption_preserved': True, 'all_global_rng_and_dropout_call_states_equal': True,
                                             'node_selection_uses_dedicated_cpu_rng': True, 'mask_nodes_per_epoch': 1354,
                                             'checks': checks, 'test_labels_used': False,
                                             'limitation': 'Equal RNG states and dropout call streams do not make floating-point GPU scatter reductions bitwise deterministic.'})
    print('M2 GPU masking/corruption/dropout RNG checks PASSED for all 5 paired seeds', flush=True)

if __name__ == '__main__':
    main()
