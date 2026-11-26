"""M2 stages: audit, pilot, validation-only training, sealed selection, testing."""
import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from run_m1 import ROOT, REPO, COMMIT, set_seed, state_hash, validation_probe, save_json, status_file, guard_gpu
from preflight_m1 import EXPECTED, digest, execute, gpu_snapshot

ALPHAS = (0.0, 0.25, 0.5, 1.0)
M1 = ROOT / 'results/raw/M1/cora/20261113T065001179740Z'

def arm(alpha):
    return 'control' if alpha == 0 else f'degree_alpha_{alpha:g}'

def load_args(seed):
    sys.path.insert(0, str(REPO))
    from graphmae.utils import build_args, load_best_configs
    previous = sys.argv
    try:
        sys.argv = ['main_transductive.py', '--dataset', 'cora', '--device', '0', '--seeds', str(seed), '--use_cfg']
        args = load_best_configs(build_args(), str(REPO / 'configs.yml'))
    finally:
        sys.argv = previous
    args.num_features = 1433
    for name, expected in EXPECTED.items():
        assert vars(args)[name] == expected, (name, vars(args)[name], expected)
    return args

def assert_inputs(batch):
    preflight = json.loads((batch / 'preflight.json').read_text())
    assert preflight['status'] == 'passed'
    assert execute(['git', '-C', REPO, 'rev-parse', 'HEAD']).strip() == COMMIT
    assert not execute(['git', '-C', REPO, 'status', '--porcelain']).strip()
    for name, value in preflight['data_sha256'].items():
        assert digest(ROOT / 'data' / name) == value
    for name, value in preflight['degree_files_sha256'].items():
        assert digest(batch / 'degree' / name) == value
    return preflight

def prepare():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    from m2_masking import calculate_degrees
    from graphmae.datasets.data_util import load_dataset
    from graphmae.models import build_model
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    batch = ROOT / 'results/raw/M2/cora' / stamp
    batch.mkdir(parents=True)
    try:
        assert execute(['git', '-C', REPO, 'rev-parse', 'HEAD']).strip() == COMMIT
        assert not execute(['git', '-C', REPO, 'status', '--porcelain']).strip()
        m1_preflight = json.loads((M1 / 'preflight.json').read_text())
        hashes = m1_preflight['data']['sha256']
        assert all(digest(ROOT / 'data' / name) == value for name, value in hashes.items())
        args = load_args(0)
        model = build_model(args)
        assert len(model.encoder.gat_layers) == 2 and len(model.decoder.gat_layers) == 1
        os.chdir(ROOT / 'data')
        graph, dimensions = load_dataset('cora')
        assert (graph.num_nodes, dimensions, graph.num_edges) == (2708, (1433, 7), 13264)
        assert [int(graph[k].sum()) for k in ['train_mask', 'val_mask', 'test_mask']] == [140, 500, 1000]
        degree_info = calculate_degrees(graph.edge_index, 2708, batch / 'degree')
        assert all(digest(ROOT / 'data' / name) == value for name, value in hashes.items())
        old_files = {}
        for directory in [ROOT / 'results/raw/M0', ROOT / 'results/raw/M1', ROOT / 'results/processed/M1', ROOT / 'figures/M1']:
            for path in directory.rglob('*'):
                if path.is_file():
                    old_files[str(path.relative_to(ROOT))] = digest(path)
        save_json(batch / 'previous_milestones_sha256.json', old_files)
        snapshots = []
        for i in range(3):
            if i:
                time.sleep(2)
            snapshots.append(gpu_snapshot())
        candidates = []
        for gpu in snapshots[0]['gpus']:
            observations = [next(g for g in snapshot['gpus'] if g['uuid'] == gpu['uuid']) for snapshot in snapshots]
            if min(g['memory.free'] for g in observations) >= 8192 and max(g['utilization.gpu'] for g in observations) < 90:
                idle = max(g['memory.used'] for g in observations) < 512
                candidates.append({'physical_index': gpu['index'], 'uuid': gpu['uuid'], 'name': gpu['name'], 'idle': idle, 'shared': not idle})
        chosen = sorted(candidates, key=lambda g: (not g['idle'], g['physical_index'] != 2, 'A40' not in g['name'], g['physical_index']))[0] if candidates else None
        report = {'stage': 'M2_preflight', 'utc': stamp, 'status': 'passed' if chosen else 'blocked_no_suitable_gpu',
                  'official_commit': COMMIT, 'data_sha256': hashes, 'official_args': vars(args),
                  'dependency_check': execute(['uv', 'pip', 'check', '--python', sys.executable]),
                  'degree': degree_info, 'degree_files_sha256': {p.name: digest(p) for p in (batch / 'degree').iterdir() if p.is_file()},
                  'gpu_snapshots': snapshots, 'selected_gpu': chosen,
                  'rng_protocol': 'Each arm uses dedicated CPU Generator(seed); control actual selection=randperm, positive actual selection=multinomial without replacement. Every arm consumes original CUDA randperm(N) first. Original replacement randperm(k) and donor randperm(N) unchanged. CPU/CUDA global RNG traced each epoch.',
                  'control_difference_from_M1': 'Actual uniform indices move from global CUDA RNG to independent CPU RNG; rerun M2 control, do not reuse or expect bitwise M1 masks.',
                  'selection_protocol': 'After all 20 validation-only runs, select positive degree_alpha by mean best validation correct count; smaller alpha on exact tie; write immutable selection before 10 final tests.',
                  'degree_alphas': list(ALPHAS), 'seeds': list(range(5)), 'training_started': False}
        save_json(batch / 'preflight.json', report)
        print(json.dumps({'batch': str(batch), 'status': report['status'], 'selected_gpu': chosen, 'degree': degree_info}, ensure_ascii=False, indent=2), flush=True)
        return 0 if chosen else 2
    except Exception:
        (batch / 'preflight_failure.txt').write_text(traceback.format_exc())
        raise

def setup_gpu(batch):
    preflight = assert_inputs(batch)
    gpu = preflight['selected_gpu']
    snapshot, current = guard_gpu(gpu['uuid'])
    if current['utilization.gpu'] >= 90:
        raise RuntimeError('Selected GPU is heavily occupied; no experiment started')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu['uuid']
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    return preflight, gpu, snapshot, current

def train_worker(batch, alpha, seed, pilot):
    preflight, gpu, snapshot, current = setup_gpu(batch)
    import numpy as np
    import torch
    from m2_masking import DegreeMasking, global_rng_hash
    sys.path.insert(0, str(REPO))
    import main_transductive as official_training
    from graphmae.datasets.data_util import load_dataset
    from graphmae.models import build_model
    from graphmae.utils import create_optimizer
    torch.set_num_threads(4)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    device = torch.device('cuda:0')
    directory = batch / ('pilot' if pilot else '') / arm(alpha) / f'seed_{seed}'
    status_file(directory / 'status.json', {'status': 'training', 'test_evaluations': 0})
    start = time.perf_counter()
    try:
        args = load_args(seed)
        if pilot:
            args.max_epoch = args.max_epoch_f = 20
        os.chdir(ROOT / 'data')
        graph, (_, classes) = load_dataset('cora')
        train_labels = graph.y[graph.train_mask].to(device)
        val_labels = graph.y[graph.val_mask].to(device)
        del graph.y
        del graph.test_mask
        graph = graph.to(device)
        train_mask, val_mask = graph.train_mask, graph.val_mask
        degrees = torch.from_numpy(np.load(batch / 'degree/degrees.npy'))
        groups = np.load(batch / 'degree/groups.npy')
        set_seed(seed)
        model = build_model(args).to(device)
        initial_hash = state_hash(model)
        before_install_rng = global_rng_hash()
        sampler = DegreeMasking(degrees, alpha, seed)
        sampler.install(model)
        assert state_hash(model) == initial_hash and global_rng_hash() == before_install_rng
        initial_token = model.enc_mask_token.detach().clone()
        optimizer = create_optimizer(args.optimizer, model, args.lr, args.weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (1 + np.cos(epoch * np.pi / args.max_epoch)) * 0.5)
        own_processes = execute(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'])
        own = [row for row in csv.reader(own_processes.splitlines(), skipinitialspace=True) if len(row) >= 2 and row[1] == str(os.getpid())]
        assert own and all(row[0] == gpu['uuid'] for row in own)
        metadata = {'stage': 'M2_pilot' if pilot else 'M2_formal', 'degree_alpha': alpha, 'seed': seed,
                    'official_args': vars(args), 'official_commit': COMMIT, 'initial_model_sha256': initial_hash,
                    'mask_rng_seed': seed, 'global_rng_after_initialization': before_install_rng,
                    'rng_protocol': preflight['rng_protocol'], 'data_sha256': preflight['data_sha256'],
                    'degree_files_sha256': preflight['degree_files_sha256'],
                    'gpu': {'physical_index': gpu['physical_index'], 'uuid': gpu['uuid'], 'logical_device': 'cuda:0', 'name': torch.cuda.get_device_name(), 'shared_at_start': current['memory.used'] > 512, 'own_process_mapping': own},
                    'cudnn_deterministic': torch.backends.cudnn.deterministic, 'cudnn_benchmark': torch.backends.cudnn.benchmark,
                    'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                    'gpu_before_start': snapshot,
                    'sources_sha256': {name: digest(ROOT / 'scripts' / name) for name in ['run_m2.py', 'm2_masking.py', 'run_m1.py']},
                    'test_policy': 'No test labels in graph or probe inputs; evaluation deferred until all 20 and sealed alpha selection.'}
        save_json(directory / 'effective_config.json', metadata)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        periodic = []
        monitors = []
        rows = []
        def periodic_probe(model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob, mute=False):
            epoch = len(rows)
            assert not pilot and epoch in range(200, 1500, 200) and linear_prob
            monitor, _ = guard_gpu(gpu['uuid'])
            monitors.append({'epoch': epoch, **monitor})
            previous_encoder = state_hash(model.encoder)
            torch.cuda.synchronize()
            began = time.perf_counter()
            classifier, _, report = validation_probe(model, graph, train_mask, val_mask, train_labels, val_labels, num_classes, lr_f, weight_decay_f, max_epoch_f, device, directory / 'probes' / f'epoch_{epoch:04d}_history.csv')
            torch.cuda.synchronize()
            report.update(pretrain_epoch=epoch, seconds=time.perf_counter() - began)
            assert state_hash(model.encoder) == previous_encoder and not report['test_evaluated']
            save_json(directory / 'probes' / f'epoch_{epoch:04d}.json', report)
            periodic.append(report)
            del classifier
            return None, None
        official_training.node_classification_evaluation = periodic_probe
        class Recorder:
            def __init__(self, handle, trace_handle):
                self.handle = handle
                self.writer = csv.DictWriter(handle, fieldnames=['epoch', 'loss', 'lr_used', 'lr_next'])
                self.writer.writeheader()
                self.trace_writer = csv.DictWriter(trace_handle, fieldnames=['epoch', 'global_cpu_after_step', 'global_cuda_after_step', 'global_cpu_after_mask', 'global_cuda_after_mask', 'replacement_permutation_sha256', 'replacement_donors_sha256'])
                self.trace_writer.writeheader()
            def note(self, values, step):
                loss = float(values['loss'])
                assert np.isfinite(loss)
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                if step == 0:
                    assert not torch.equal(initial_token, model.enc_mask_token.detach())
                    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
                assert sampler.calls == step + 1 and sampler.last_trace['selected_count'] == 1354
                row = {'epoch': step + 1, 'loss': loss, 'lr_used': args.lr * (1 + np.cos(step * np.pi / args.max_epoch)) * 0.5, 'lr_next': values['lr']}
                rows.append(row)
                self.writer.writerow(row)
                cpu, cuda = global_rng_hash()
                self.trace_writer.writerow({'epoch': step + 1, 'global_cpu_after_step': cpu, 'global_cuda_after_step': cuda, **{key: sampler.last_trace[key] for key in ['global_cpu_after_mask', 'global_cuda_after_mask', 'replacement_permutation_sha256', 'replacement_donors_sha256']}})
                if (step + 1) % 50 == 0:
                    handle.flush()
                    trace_handle.flush()
        (directory / 'probes').mkdir()
        torch.cuda.synchronize()
        pretrain_start = time.perf_counter()
        with (directory / 'training.csv').open('x', newline='') as handle, (directory / 'rng_trace.csv').open('x', newline='') as trace_handle:
            recorder = Recorder(handle, trace_handle)
            model = official_training.pretrain(model, graph, graph.x, optimizer, args.max_epoch, device, scheduler, classes, args.lr_f, args.weight_decay_f, args.max_epoch_f, args.linear_prob, recorder)
        torch.cuda.synchronize()
        pretrain_seconds = time.perf_counter() - pretrain_start
        assert len(rows) == args.max_epoch and sampler.calls == args.max_epoch
        assert len(periodic) == (0 if pilot else 7)
        assert all(torch.isfinite(p).all() for p in model.parameters())
        model = model.cpu()
        torch.save({'state_dict': model.state_dict(), 'official_args': vars(args), 'seed': seed, 'degree_alpha': alpha, 'epoch': args.max_epoch, 'commit': COMMIT}, directory / 'pretrained_final.pt')
        model = model.to(device)
        encoder_before = state_hash(model.encoder)
        torch.cuda.synchronize()
        began = time.perf_counter()
        best_classifier, _, selection = validation_probe(model, graph, train_mask, val_mask, train_labels, val_labels, classes, args.lr_f, args.weight_decay_f, args.max_epoch_f, device, directory / 'probe_history.csv')
        torch.cuda.synchronize()
        probe_seconds = time.perf_counter() - began
        assert state_hash(model.encoder) == encoder_before and not selection['test_evaluated']
        save_json(directory / 'classifier_selection.json', selection)
        torch.save({'state_dict': {key: value.detach().cpu() for key, value in best_classifier.state_dict().items()}, 'selection': selection, 'seed': seed, 'degree_alpha': alpha, 'input_dim': 512, 'classes': 7}, directory / 'classifier_best_val.pt')
        stats = sampler.save_statistics(directory, groups)
        save_json(directory / 'periodic_probes.json', periodic)
        save_json(directory / 'gpu_monitor.json', monitors)
        report = {'status': 'validation_complete', 'pilot': pilot, 'seed': seed, 'degree_alpha': alpha,
                  'initial_model_sha256': initial_hash, 'best_val_accuracy': selection['best_val_accuracy'],
                  'best_val_correct': round(selection['best_val_accuracy'] * 500),
                  'selected_epoch': selection['selected_epoch'], 'pretrain_epochs': args.max_epoch,
                  'probe_epochs': args.max_epoch_f, 'periodic_probes': len(periodic),
                  'test_evaluations': 0, 'test_deferred': True,
                  'first_loss': rows[0]['loss'], 'last_loss': rows[-1]['loss'],
                  'pretrain_seconds': pretrain_seconds - sum(p['seconds'] for p in periodic),
                  'periodic_probe_seconds': sum(p['seconds'] for p in periodic), 'final_probe_seconds': probe_seconds,
                  'total_seconds': time.perf_counter() - start, 'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
                  'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20, 'gpu': metadata['gpu'], 'mask_groups': stats,
                  'checkpoint_sha256': {name: digest(directory / name) for name in ['pretrained_final.pt', 'classifier_best_val.pt']}}
        save_json(directory / 'result.json', report)
        status_file(directory / 'status.json', {'status': 'validation_complete', 'test_evaluations': 0})
        print(json.dumps(report, ensure_ascii=False), flush=True)
    except Exception:
        (directory / 'failure.txt').write_text(traceback.format_exc())
        status_file(directory / 'status.json', {'status': 'failed', 'test_evaluations': 0})
        raise

def validate_training(directory, pilot=False):
    report = json.loads((directory / 'result.json').read_text())
    assert report['status'] == 'validation_complete' and report['test_evaluations'] == 0
    assert report['pretrain_epochs'] == (20 if pilot else 1500)
    assert report['probe_epochs'] == (20 if pilot else 300)
    assert report['periodic_probes'] == (0 if pilot else 7)
    with (directory / 'training.csv').open() as handle:
        assert len(list(csv.DictReader(handle))) == report['pretrain_epochs']
    with (directory / 'probe_history.csv').open() as handle:
        history = list(csv.DictReader(handle))
    assert len(history) == report['probe_epochs']
    best = max(float(row['val_accuracy']) for row in history)
    epoch = max(int(row['epoch']) for row in history if float(row['val_accuracy']) == best)
    assert best == report['best_val_accuracy'] and epoch == report['selected_epoch']
    for name, expected in report['checkpoint_sha256'].items():
        assert digest(directory / name) == expected
    return report

def check_paired_rng(batch, pilot):
    seeds = [0] if pilot else range(5)
    summary = []
    for seed in seeds:
        initial = []
        traces = []
        reports = []
        for alpha in ALPHAS:
            directory = batch / ('pilot' if pilot else '') / arm(alpha) / f'seed_{seed}'
            reports.append(validate_training(directory, pilot))
            initial.append(reports[-1]['initial_model_sha256'])
            with (directory / 'rng_trace.csv').open() as handle:
                traces.append(list(csv.DictReader(handle)))
        assert len(set(initial)) == 1
        assert all(trace == traces[0] for trace in traces[1:]), 'Global RNG or donor/dropout random consumption differs between arms'
        high_rates = [report['mask_groups'][2]['mask_rate_within_group'] for report in reports]
        if pilot:
            assert all(b > a for a, b in zip(high_rates, high_rates[1:])), high_rates
        summary.append({'seed': seed, 'initial_model_sha256': initial[0], 'initial_weights_equal': True, 'all_epoch_global_rng_and_replacement_traces_equal': True, 'high_degree_mask_rates': high_rates})
    return summary

def launch_runs(batch, pilot):
    assert_inputs(batch)
    assert json.loads((batch / 'masking_checks.json').read_text())['status'] == 'passed'
    if not pilot:
        assert json.loads((batch / 'pilot_check.json').read_text())['status'] == 'passed'
    for seed in ([0] if pilot else range(5)):
        for alpha in ALPHAS:
            directory = batch / ('pilot' if pilot else '') / arm(alpha) / f'seed_{seed}'
            if directory.exists():
                validate_training(directory, pilot)  # completed-only recovery; never overwrite partial runs
                continue
            directory.mkdir(parents=True)
            command = [sys.executable, str(Path(__file__).resolve()), '--stage', 'worker', '--batch', str(batch), '--seed', str(seed), '--degree-alpha', str(alpha)]
            if pilot:
                command.append('--pilot')
            with (directory / 'run.log').open('x') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                print((directory / 'run.log').read_text()[-8000:], flush=True)
                raise RuntimeError('Run failed; remaining runs stopped: ' + str(directory))
            report = validate_training(directory, pilot)
            print(json.dumps({'pilot': pilot, 'seed': seed, 'degree_alpha': alpha, 'best_val_accuracy': report['best_val_accuracy'], 'selected_epoch': report['selected_epoch'], 'seconds': report['total_seconds'], 'test_evaluations': 0}, ensure_ascii=False), flush=True)
    paired = check_paired_rng(batch, pilot)
    save_json(batch / ('pilot_check.json' if pilot else 'formal_training_check.json'), {'status': 'passed', 'paired_rng_checks': paired, 'test_evaluations': 0})

def select_alpha(batch):
    assert json.loads((batch / 'formal_training_check.json').read_text())['status'] == 'passed'
    check_paired_rng(batch, False)
    groups = []
    input_hashes = {}
    for alpha in ALPHAS:
        reports = []
        for seed in range(5):
            directory = batch / arm(alpha) / f'seed_{seed}'
            reports.append(validate_training(directory))
            assert not (directory / 'test_started.json').exists()
            input_hashes[str(directory.relative_to(batch))] = {name: digest(directory / name) for name in ['result.json', 'classifier_selection.json', 'pretrained_final.pt', 'classifier_best_val.pt']}
        groups.append({'degree_alpha': alpha, 'validation_correct_sum': sum(r['best_val_correct'] for r in reports), 'validation_accuracy_mean': sum(r['best_val_correct'] for r in reports) / 2500, 'seed_validation_accuracy': [r['best_val_accuracy'] for r in reports]})
    selected = min([g for g in groups if g['degree_alpha'] > 0], key=lambda g: (-g['validation_correct_sum'], g['degree_alpha']))
    save_json(batch / 'alpha_selection.json', {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'selected_degree_alpha': selected['degree_alpha'], 'criterion': 'maximum five-seed mean best validation accuracy among positive coefficients; tie choose smaller alpha, using exact integer correct totals', 'groups': groups, 'inputs_sha256': input_hashes, 'test_results_available_at_selection': False, 'all_20_training_complete': True})
    print('SEALED ALPHA SELECTION', json.dumps(groups), 'selected=', selected['degree_alpha'], flush=True)

def test_worker(batch, alpha, seed):
    selection_path = batch / 'alpha_selection.json'
    selection = json.loads(selection_path.read_text())
    assert alpha in [0.0, selection['selected_degree_alpha']]
    preflight, gpu, _, _ = setup_gpu(batch)
    import numpy as np
    import torch
    from sklearn.metrics import accuracy_score, f1_score
    from graphmae.datasets.data_util import load_dataset
    from graphmae.models import build_model
    from graphmae.evaluation import LogisticRegression
    torch.set_num_threads(4)
    directory = batch / arm(alpha) / f'seed_{seed}'
    report = validate_training(directory)
    for name, value in selection['inputs_sha256'][str(directory.relative_to(batch))].items():
        assert digest(directory / name) == value
    selection_hash = digest(selection_path)
    save_json(directory / 'test_started.json', {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'selection_sha256': selection_hash, 'degree_alpha': alpha, 'seed': seed})
    os.chdir(ROOT / 'data')
    graph, _ = load_dataset('cora')
    args = load_args(seed)
    model = build_model(args).to('cuda:0')
    model.load_state_dict(torch.load(directory / 'pretrained_final.pt', map_location='cuda:0', weights_only=False)['state_dict'])
    classifier = LogisticRegression(512, 7).to('cuda:0')
    classifier.load_state_dict(torch.load(directory / 'classifier_best_val.pt', map_location='cuda:0', weights_only=False)['state_dict'])
    model.eval()
    classifier.eval()
    with torch.no_grad():
        embedding = model.embed(graph.x.to('cuda:0'), graph.edge_index.to('cuda:0'))
        logits = classifier(None, embedding)
        prediction = logits[graph.test_mask.to('cuda:0')].argmax(1).cpu().numpy()
    # Test labels are first used here, after all validation-only selection is sealed.
    target = graph.y[graph.test_mask].numpy()
    accuracy = float(accuracy_score(target, prediction))
    micro = float(f1_score(target, prediction, average='micro'))
    assert accuracy == micro and digest(selection_path) == selection_hash
    np.savez_compressed(directory / 'test_predictions.npz', node_id=graph.test_mask.nonzero(as_tuple=True)[0].numpy(), target=target, prediction=prediction)
    save_json(directory / 'final_test.json', {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'seed': seed, 'degree_alpha': alpha, 'test_accuracy': accuracy, 'test_micro_f1': micro, 'test_nodes': 1000, 'test_evaluations': 1, 'selection_sha256': selection_hash, 'test_after_alpha_selection': True, 'prediction_sha256': digest(directory / 'test_predictions.npz')})
    print(json.dumps({'seed': seed, 'degree_alpha': alpha, 'test_accuracy': accuracy, 'test_evaluations': 1}), flush=True)

def final_tests(batch):
    selection = json.loads((batch / 'alpha_selection.json').read_text())
    for alpha in [0.0, selection['selected_degree_alpha']]:
        for seed in range(5):
            directory = batch / arm(alpha) / f'seed_{seed}'
            if (directory / 'final_test.json').exists():
                continue
            if (directory / 'test_started.json').exists():
                raise RuntimeError('Partial previous test exists; refusing to evaluate this model again')
            command = [sys.executable, str(Path(__file__).resolve()), '--stage', 'test-worker', '--batch', str(batch), '--seed', str(seed), '--degree-alpha', str(alpha)]
            with (directory / 'test.log').open('x') as handle:
                result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
            if result.returncode:
                print((directory / 'test.log').read_text()[-6000:], flush=True)
                raise RuntimeError('Final test failed; stopping')
            print((directory / 'final_test.json').read_text(), flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['prepare', 'pilot', 'train', 'select', 'test', 'worker', 'test-worker'], required=True)
    parser.add_argument('--batch', type=Path)
    parser.add_argument('--degree-alpha', type=float, choices=ALPHAS)
    parser.add_argument('--seed', type=int, choices=range(5))
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(REPO))
    if args.stage == 'prepare':
        return prepare()
    assert args.batch and args.batch.resolve().is_relative_to(ROOT / 'results/raw/M2/cora')
    batch = args.batch.resolve()
    if args.stage == 'worker':
        train_worker(batch, args.degree_alpha, args.seed, args.pilot)
    elif args.stage == 'test-worker':
        test_worker(batch, args.degree_alpha, args.seed)
    elif args.stage in ['pilot', 'train']:
        launch_runs(batch, args.stage == 'pilot')
    elif args.stage == 'select':
        select_alpha(batch)
    else:
        final_tests(batch)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
