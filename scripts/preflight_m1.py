"""Audit formal Cora settings and resources without starting any training."""
import csv
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / 'third_party/GraphMAE'
COMMIT = '9127f25e01aa5ba1cca68b746284aeb7ab2b4850'
EXPECTED = {
    'dataset': 'cora', 'num_features': 1433, 'num_hidden': 512,
    'num_heads': 4, 'num_out_heads': 1, 'num_layers': 2,
    'encoder': 'gat', 'decoder': 'gat', 'mask_rate': 0.5,
    'replace_rate': 0.05, 'loss_fn': 'sce', 'alpha_l': 3,
    'optimizer': 'adam', 'lr': 0.001, 'weight_decay': 0.0002,
    'in_drop': 0.2, 'attn_drop': 0.1, 'drop_edge_rate': 0.0,
    'scheduler': True, 'max_epoch': 1500, 'max_epoch_f': 300,
    'lr_f': 0.01, 'weight_decay_f': 0.0001, 'linear_prob': True,
    'activation': 'prelu', 'residual': False, 'norm': None,
    'concat_hidden': False,
}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def execute(command):
    result = subprocess.run(list(map(str, command)), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    if result.returncode:
        raise RuntimeError(str(command) + '\n' + result.stdout)
    return result.stdout

def write_json(path, value):
    with path.open('x') as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write('\n')

def gpu_snapshot():
    fields = ['index', 'name', 'uuid', 'memory.total', 'memory.free', 'memory.used', 'utilization.gpu']
    result = execute(['nvidia-smi', '--query-gpu=' + ','.join(fields), '--format=csv,noheader,nounits'])
    gpus = []
    for values in csv.reader(result.splitlines(), skipinitialspace=True):
        gpu = dict(zip(fields, values))
        for key in ['index', 'memory.total', 'memory.free', 'memory.used', 'utilization.gpu']:
            gpu[key] = int(gpu[key])
        gpus.append(gpu)
    return {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'gpus': gpus, 'full_nvidia_smi': execute(['nvidia-smi'])}

def main():
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    directory = ROOT / 'results/raw/M1/cora' / stamp
    directory.mkdir(parents=True)
    report = {'stage': 'M1_preflight', 'utc': stamp, 'training_started': False, 'batch': str(directory), 'checks': {}}
    write_json(directory / 'status.json', {'status': 'preflight_running', 'training_started': False})
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # no GPU allocations in preflight
        os.environ.setdefault('OMP_NUM_THREADS', '4')
        import numpy as np
        import torch
        import yaml
        sys.path.insert(0, str(REPO))
        from graphmae.utils import build_args, load_best_configs
        from graphmae.models import build_model
        from graphmae.datasets.data_util import load_dataset
        report['commit'] = execute(['git', '-C', REPO, 'rev-parse', 'HEAD']).strip()
        report['git_status'] = execute(['git', '-C', REPO, 'status', '--porcelain']).strip()
        assert report['commit'] == COMMIT and not report['git_status']
        report['checks']['official_code'] = 'passed'
        report['official_files_sha256'] = {name: digest(REPO / name) for name in ['configs.yml', 'main_transductive.py', 'graphmae/evaluation.py', 'graphmae/utils.py', 'graphmae/models/edcoder.py', 'graphmae/models/gat.py', 'graphmae/models/loss_func.py', 'graphmae/datasets/data_util.py']}
        packages = ['torch', 'torch-geometric', 'torch-scatter', 'torch-sparse', 'numpy', 'scipy', 'scikit-learn', 'PyYAML', 'ogb', 'tensorboardX']
        report['environment'] = {'python': sys.version, 'executable': sys.executable, 'torch_cuda_runtime': torch.version.cuda, 'packages': {p: importlib.metadata.version(p) for p in packages}}
        assert sys.version_info[:2] == (3, 11)
        assert torch.__version__ == '2.4.1+cu121'
        assert importlib.metadata.version('torch-geometric') == '2.3.1'
        report['dependency_check'] = execute(['uv', 'pip', 'check', '--python', sys.executable])
        report['checks']['environment'] = 'passed'
        old_args = sys.argv
        sys.argv = ['main_transductive.py', '--dataset', 'cora', '--seeds', '0', '1', '2', '3', '4', '--device', '0', '--use_cfg']
        args = load_best_configs(build_args(), str(REPO / 'configs.yml'))
        sys.argv = old_args
        args.num_features = 1433
        config = vars(args)
        for name, expected in EXPECTED.items():
            assert config[name] == expected, (name, config[name], expected)
        assert config['seeds'] == [0, 1, 2, 3, 4]
        yaml_tree = yaml.compose((REPO / 'configs.yml').read_text())
        cora = next(value for key, value in yaml_tree.value if key.value == 'cora')
        values = {}
        for key, value in cora.value:
            values.setdefault(key.value, []).append(value.value)
        duplicates = {key: value for key, value in values.items() if len(value) > 1}
        assert all(len(set(values)) == 1 for values in duplicates.values()), duplicates
        report['yaml_duplicate_same_value_keys'] = duplicates
        model = build_model(args)
        assert len(model.encoder.gat_layers) == 2
        assert len(model.decoder.gat_layers) == 1
        report['architecture'] = {'encoder_layers': 2, 'decoder_layers': 1, 'parameter_count': sum(p.numel() for p in model.parameters()), 'repr': repr(model)}
        report['checks']['official_effective_config_and_architecture'] = 'passed'
        write_json(directory / 'effective_config.json', {'official_args': config, 'decoder_num_layers_verified': 1, 'scope': 'formal Cora baseline; no degree masking or Citeseer', 'rng_settings_planned': {'python_numpy_torch_cpu_cuda': 'seed per independent process', 'cudnn_deterministic': True, 'cudnn_benchmark': False, 'torch_use_deterministic_algorithms': False}, 'evaluation_protocol': {'periodic_epochs': list(range(200, 1500, 200)), 'periodic_probe_epochs': 300, 'final_probe_epochs': 300, 'probe_selection': 'validation accuracy; >= retains later tied epoch', 'pretrained_encoder_selection': 'fixed epoch 1500', 'test_calls_per_seed': 1, 'test_used_for_selection': False}})
        m0 = ROOT / 'results/raw/M0/smoke_20261112T040413960326Z'
        old_summary = json.loads((m0 / 'summary.json').read_text())
        hashes = {name: digest(ROOT / 'data' / name) for name in old_summary['data_sha256']}
        assert hashes == old_summary['data_sha256']
        smoke_config = json.loads((m0 / 'effective_config.json').read_text())
        differences = {name: {'M0': smoke_config.get(name), 'M1': value} for name, value in config.items() if smoke_config.get(name) != value}
        assert set(differences) == {'max_epoch', 'max_epoch_f', 'seeds', 'device', 'use_cfg'}, differences
        report['M0_to_M1_config_differences'] = differences
        report['M0_smoke_script_sha256'] = digest(ROOT / 'scripts/smoke_test.py')
        report['M0_smoke_config_sha256'] = digest(ROOT / 'configs/smoke.json')
        os.chdir(ROOT / 'data')
        graph, (features, classes) = load_dataset('cora')
        assert (graph.num_nodes, features, classes, graph.num_edges) == (2708, 1433, 7, 13264)
        masks = [graph[name] for name in ['train_mask', 'val_mask', 'test_mask']]
        assert [int(mask.sum()) for mask in masks] == [140, 500, 1000]
        assert not any((masks[i] & masks[j]).any() for i, j in [(0, 1), (0, 2), (1, 2)])
        assert torch.isfinite(graph.x).all()
        sums = graph.x.sum(1)
        assert torch.allclose(sums[sums > 0], torch.ones_like(sums[sums > 0]), atol=1e-6)
        self_loops = graph.edge_index[:, graph.edge_index[0] == graph.edge_index[1]]
        assert self_loops.shape[1] == 2708 and self_loops[0].unique().numel() == 2708
        assert torch.unique(graph.edge_index.T, dim=0).shape[0] == 13264
        assert {name: digest(ROOT / 'data' / name) for name in hashes} == hashes
        report['data'] = {'nodes': 2708, 'features': 1433, 'classes': 7, 'directed_edge_entries_including_self_loops': 13264, 'split': [140, 500, 1000], 'normalize_features_verified': True, 'one_self_loop_per_node': True, 'sha256_matches_M0_before_and_after_load': True, 'sha256': hashes}
        report['checks']['M0_data_and_preprocessing'] = 'passed'
        report['gpu_mapping_audit'] = {'M0_smoke_cli_gpu': 'physical nvidia-smi index; sets CUDA_VISIBLE_DEVICES before importing torch', 'M0_success_physical': old_summary['physical_gpu'], 'M0_success_internal_device': smoke_config['device'], 'README_gpu_1': 'example only, not historical actual selection', 'planned_M1_mapping': 'resolve physical nvidia-smi index to UUID, set CUDA_VISIBLE_DEVICES=UUID and use internal cuda:0; record both', 'preflight_cuda_visible_devices': '', 'note': 'Preflight intentionally performs no GPU tensor allocation.'}
        snapshots = []
        for index in range(3):
            if index:
                time.sleep(3)
            snapshots.append(gpu_snapshot())
        report['gpu_snapshots'] = snapshots
        candidates = []
        eligibility = []
        for gpu in snapshots[0]['gpus']:
            samples = [next(g for g in snapshot['gpus'] if g['uuid'] == gpu['uuid']) for snapshot in snapshots]
            free = min(sample['memory.free'] for sample in samples)
            utilization = [sample['utilization.gpu'] for sample in samples]
            idle = max(sample['memory.used'] for sample in samples) < 512 and max(utilization) < 10
            # Do not launch into a device that is already at sustained full load.
            eligible = free >= 8192 and max(utilization) < 90
            item = {'physical_index': gpu['index'], 'uuid': gpu['uuid'], 'name': gpu['name'], 'minimum_free_mib': free, 'utilization_samples_pct': utilization, 'idle': idle, 'shared': not idle, 'eligible': eligible}
            eligibility.append(item)
            if eligible:
                candidates.append(item)
        report['gpu_eligibility'] = eligibility
        report['gpu_selection_rule'] = 'prefer idle GPU; shared requires >=8192 MiB free and utilization below 90% in each of three observations; never terminate other processes'
        if candidates:
            selected = sorted(candidates, key=lambda g: (not g['idle'], max(g['utilization_samples_pct']), -g['minimum_free_mib']))[0]
            report['selected_gpu'] = selected
            report['status'] = 'passed'
            report['checks']['gpu'] = 'passed'
        else:
            report['selected_gpu'] = None
            report['status'] = 'blocked_no_suitable_gpu'
            report['checks']['gpu'] = 'blocked: occupied or insufficient free memory'
        write_json(directory / 'preflight.json', report)
        write_json(directory / 'final_status.json', {'status': report['status'], 'training_started': False, 'seeds_completed': [], 'formal_experiment_results': None})
        print(json.dumps({'batch': str(directory), 'status': report['status'], 'checks': report['checks'], 'gpus': eligibility, 'training_started': False}, ensure_ascii=False, indent=2), flush=True)
        return 0 if candidates else 2
    except Exception:
        report.update(status='failed_preflight', traceback=traceback.format_exc())
        write_json(directory / 'preflight_failed.json', report)
        raise

if __name__ == '__main__':
    raise SystemExit(main())
