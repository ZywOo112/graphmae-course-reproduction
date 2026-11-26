"""Bounded M0 test of the unmodified official PyG GraphMAE path."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / 'third_party/GraphMAE'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=1, help='Physical GPU index; -1 for CPU')
    parser.add_argument('--allow-shared', action='store_true', help='Allow sharing only with at least 8 GiB free')
    cli = parser.parse_args()
    config = json.loads((ROOT / 'configs/smoke.json').read_text())
    assert 1 <= config['pretrain_epochs'] <= 30
    assert 1 <= config['probe_epochs'] <= 50
    if cli.gpu >= 0:
        query = subprocess.check_output(['nvidia-smi', f'--id={cli.gpu}', '--query-gpu=memory.used,memory.free', '--format=csv,noheader,nounits'], text=True)
        used_mib, free_mib = map(int, query.strip().split(','))
        if free_mib < 8192:
            raise SystemExit('Chosen GPU has less than 8 GiB free.')
        if used_mib > 1024 and not cli.allow_shared:
            raise SystemExit('Chosen GPU is occupied (>1 GiB); select an idle GPU.')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cli.gpu) if cli.gpu >= 0 else ''
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = ROOT / 'results/raw/M0' / ('smoke_' + stamp)
    output.mkdir()
    data_root = ROOT / 'data'
    data_root.mkdir(exist_ok=True)
    os.chdir(data_root)  # official Planetoid root is relative to CWD
    sys.path.insert(0, str(REPO))
    import numpy as np
    import torch
    from graphmae.datasets.data_util import load_dataset
    from graphmae.models import build_model
    from graphmae.utils import build_args, load_best_configs, set_random_seed, create_optimizer
    from graphmae.evaluation import node_classification_evaluation
    from main_transductive import pretrain

    torch.set_num_threads(4)
    device = torch.device('cuda:0' if cli.gpu >= 0 else 'cpu')
    if cli.gpu >= 0:
        assert torch.cuda.is_available()
    def sync():
        if cli.gpu >= 0:
            torch.cuda.synchronize()
    sys.argv = [str(REPO / 'main_transductive.py'), '--dataset', config['dataset']]
    args = load_best_configs(build_args(), str(REPO / 'configs.yml'))
    args.max_epoch = config['pretrain_epochs']
    args.max_epoch_f = config['probe_epochs']
    args.seeds = [config['seed']]
    args.device = str(device)
    start = time.perf_counter()
    graph, (features, classes) = load_dataset(args.dataset)
    dataset_seconds = time.perf_counter() - start
    graph = graph.to(device)
    args.num_features = features
    (output / 'effective_config.json').write_text(json.dumps(vars(args), indent=2) + '\n')
    assert graph.num_nodes == 2708 and features == 1433 and classes == 7
    assert [int(graph[k].sum()) for k in ('train_mask', 'val_mask', 'test_mask')] == [140, 500, 1000]
    assert not (graph.train_mask & graph.val_mask).any()
    assert not (graph.train_mask & graph.test_mask).any()
    assert not (graph.val_mask & graph.test_mask).any()
    assert torch.isfinite(graph.x).all()
    nonzero = graph.x.sum(1) > 0
    assert torch.allclose(graph.x.sum(1)[nonzero], torch.ones_like(graph.x.sum(1)[nonzero]), atol=1e-6)
    loops = graph.edge_index[0] == graph.edge_index[1]
    assert int(loops.sum()) == graph.num_nodes
    set_random_seed(config['seed'])
    model = build_model(args).to(device)
    model.train()
    before = model.enc_mask_token.detach().clone()
    use_x, (masked, kept) = model.encoding_mask_noise(graph.x, args.mask_rate)
    assert masked.numel() == 1354 and masked.unique().numel() == 1354
    assert torch.equal(use_x[kept], graph.x[kept])
    loss, _ = model(graph.x, graph.edge_index)
    assert torch.isfinite(loss) and loss.ndim == 0
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(float(g.abs().sum()) for g in grads) > 0
    forward_loss = float(loss.detach())
    with torch.no_grad():
        embedding = model.embed(graph.x, graph.edge_index)
        assert tuple(embedding.shape) == (2708, 512)
        assert torch.isfinite(embedding).all()

    # Reinitialize after diagnostics, keeping the training RNG sequence explicit.
    set_random_seed(config['seed'])
    model = build_model(args).to(device)
    optimizer = create_optimizer(args.optimizer, model, args.lr, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (1 + np.cos(epoch * np.pi / args.max_epoch)) * 0.5)
    class Recorder:
        def __init__(self):
            self.rows = []
            sync()
            self.previous = time.perf_counter()
        def note(self, values, step):
            sync()
            now = time.perf_counter()
            self.rows.append({'epoch': step, **values, 'seconds': now - self.previous})
            self.previous = now
    recorder = Recorder()
    sync()
    start = time.perf_counter()
    model = pretrain(model, graph, graph.x, optimizer, args.max_epoch, device, scheduler, classes, args.lr_f, args.weight_decay_f, args.max_epoch_f, args.linear_prob, recorder)
    sync()
    pretrain_seconds = time.perf_counter() - start
    assert len(recorder.rows) == args.max_epoch
    assert all(np.isfinite(row['loss']) for row in recorder.rows)
    assert not torch.equal(model.enc_mask_token.detach(), before)
    encoder_before_probe = {name: tensor.detach().clone() for name, tensor in model.encoder.state_dict().items()}
    start = time.perf_counter()
    final_acc, best_val_test_acc = node_classification_evaluation(model, graph, graph.x, classes, args.lr_f, args.weight_decay_f, args.max_epoch_f, device, args.linear_prob, mute=False)
    sync()
    probe_seconds = time.perf_counter() - start
    assert all(torch.equal(tensor, model.encoder.state_dict()[name]) for name, tensor in encoder_before_probe.items())
    data_hashes = {str(p.relative_to(data_root)): digest(p) for p in sorted(data_root.rglob('*')) if p.is_file()}
    summary = {
        'stage': 'M0_smoke_only', 'status': 'passed', 'is_formal_experiment': False,
        'utc': stamp, 'commit': subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
        'python': sys.version, 'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
        'physical_gpu': cli.gpu, 'gpu_name': torch.cuda.get_device_name() if cli.gpu >= 0 else 'CPU',
        'shared_gpu': cli.allow_shared, 'gpu_existing_used_mib': used_mib if cli.gpu >= 0 else None,
        'nodes': graph.num_nodes, 'features': features, 'classes': classes,
        'directed_edges_with_self_loops': graph.num_edges, 'self_loops': int(loops.sum()),
        'split_sizes': [140, 500, 1000], 'embedding_shape': list(embedding.shape),
        'masked_nodes': int(masked.numel()), 'forward_loss': forward_loss,
        'pretrain_epochs': args.max_epoch, 'probe_epochs': args.max_epoch_f,
        'first_train_loss': recorder.rows[0]['loss'], 'last_train_loss': recorder.rows[-1]['loss'],
        'final_probe_accuracy': final_acc, 'validation_selected_probe_test_accuracy': best_val_test_acc,
        'encoder_unchanged_by_probe': True, 'dataset_seconds': dataset_seconds,
        'pretrain_seconds': pretrain_seconds, 'probe_seconds': probe_seconds,
        'median_epoch_seconds_excluding_first_two': statistics.median(r['seconds'] for r in recorder.rows[2:]),
        'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20 if cli.gpu >= 0 else None,
        'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20 if cli.gpu >= 0 else None,
        'data_sha256': data_hashes,
        'note': '20-epoch smoke metrics are diagnostic only, not evidence of reproduction or improvement.'
    }
    (output / 'epochs.json').write_text(json.dumps(recorder.rows, indent=2) + '\n')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)
    print('Artifacts:', output, flush=True)

if __name__ == '__main__':
    main()
