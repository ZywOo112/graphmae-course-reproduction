"""Check probe-update and RNG equivalence on a toy fixture, not a Cora run."""
import datetime
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
from run_m1 import ROOT, REPO, validation_probe, save_json, digest

def main():
    import torch
    from torch_geometric.data import Data
    sys.path.insert(0, str(REPO))
    import graphmae.evaluation as official
    torch.set_num_threads(1)
    torch.manual_seed(41)
    x = torch.randn(18, 5)
    labels = torch.arange(18) % 3
    train = torch.arange(18) < 8
    val = (torch.arange(18) >= 8) & (torch.arange(18) < 12)
    test = torch.arange(18) >= 12
    graph = Data(x=x, edge_index=torch.empty((2, 0), dtype=torch.long), y=labels,
                 train_mask=train, val_mask=val, test_mask=test)
    class FixedEmbedding(torch.nn.Module):
        def embed(self, features, edge_index):
            return features
    model = FixedEmbedding()
    states = []
    class ObservedClassifier(official.LogisticRegression):
        def forward(self, graph, features, *args):
            result = super().forward(graph, features, *args)
            states.append({key: value.detach().clone() for key, value in self.state_dict().items()})
            return result
    original_class = official.LogisticRegression
    original_accuracy = official.accuracy
    val_history = []
    def observed_accuracy(prediction, target):
        value = original_accuracy(prediction, target)
        if len(target) == 4:
            val_history.append(value)
        return value
    initial_rng = torch.get_rng_state().clone()
    try:
        official.LogisticRegression = ObservedClassifier
        official.accuracy = observed_accuracy
        official.node_classification_evaluation(model, graph.clone(), x, 3, 0.01, 0.0001, 30, 'cpu', True, True)
        reference_best = states[-1]
        reference_final = states[-2]
        reference_rng = torch.get_rng_state().clone()
        assert len(val_history) == 30
        best_val = max(val_history)
        best_epoch = max(index + 1 for index, value in enumerate(val_history) if value == best_val)
        official.LogisticRegression = original_class
        official.accuracy = original_accuracy
        sanitized = graph.clone()
        del sanitized.y
        del sanitized.test_mask
        torch.set_rng_state(initial_rng)
        selected, _, result = validation_probe(model, sanitized, train, val, labels[train], labels[val], 3, 0.01, 0.0001, 30, 'cpu')
        assert all(torch.equal(reference_best[name], value) for name, value in selected.state_dict().items())
        assert result['best_val_accuracy'] == best_val and result['selected_epoch'] == best_epoch
        assert torch.equal(reference_rng, torch.get_rng_state())
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        output = ROOT / 'results/raw/M1' / ('protocol_check_' + stamp + '.json')
        save_json(output, {'status': 'passed', 'fixture': '18 synthetic nodes, 5 embedding dimensions, 30 probe epochs; not Cora experiment data', 'official_and_adapter_selected_weights_identical': True, 'official_and_adapter_best_val_and_tie_epoch_identical': True, 'cpu_torch_rng_after_probe_identical': True, 'adapter_graph_has_no_labels_or_test_mask': True, 'adapter_does_not_accept_test_targets': True, 'best_val_accuracy_on_toy_fixture': best_val, 'selected_toy_probe_epoch': best_epoch, 'source_sha256': {'run_m1.py': digest(ROOT / 'scripts/run_m1.py')}, 'limitation': 'CPU fixture proves probe algorithm/RNG equivalence; GPU scatter can remain nondeterministic.'})
        print('Probe protocol equivalence check passed:', output, flush=True)
    finally:
        official.LogisticRegression = original_class
        official.accuracy = original_accuracy

if __name__ == '__main__':
    main()
