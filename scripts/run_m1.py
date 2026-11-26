"""Run official GraphMAE pretraining with validation-only periodic probes."""
import argparse
import copy
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

from preflight_m1 import ROOT, REPO, COMMIT, EXPECTED, digest, execute, gpu_snapshot

def save_json(path, value):
    with path.open('x') as output:
        json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write('\n')

def status_file(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)

def set_seed(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Deliberately do not force deterministic GPU scatter algorithms.

def validation_probe(model, graph, train_mask, val_mask, train_labels,
                     val_labels, num_classes, lr, weight_decay, epochs,
                     device, history_path=None):
    """Official linear-probe updates and >= tie rule; no test targets accepted."""
    import torch
    from graphmae.evaluation import LogisticRegression
    from graphmae.utils import create_optimizer, accuracy
    model.eval()
    with torch.no_grad():
        embedding = model.embed(graph.x.to(device), graph.edge_index.to(device))
    assert not embedding.requires_grad
    probe = LogisticRegression(embedding.shape[1], num_classes)
    probe.to(device)
    optimizer = create_optimizer('adam', probe, lr, weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    best_accuracy = 0
    best_epoch = 0
    best_model = None
    rows = []
    for epoch in range(epochs):
        probe.train()
        out = probe(graph, embedding)
        loss = criterion(out[train_mask], train_labels)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite linear probe loss')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            probe.eval()
            prediction = probe(graph, embedding)
            val_accuracy = accuracy(prediction[val_mask], val_labels)
            val_loss = criterion(prediction[val_mask], val_labels)
        if not torch.isfinite(val_loss):
            raise RuntimeError('Non-finite validation loss')
        if val_accuracy >= best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch
            best_model = copy.deepcopy(probe)
        rows.append({'epoch': epoch + 1, 'epoch_zero_based': epoch,
                     'train_loss': float(loss.detach()), 'val_loss': float(val_loss),
                     'val_accuracy': val_accuracy})
    best_model.eval()
    with torch.no_grad():
        # Preserve the final official best-classifier forward, but use validation.
        selected_prediction = best_model(graph, embedding)
        assert accuracy(selected_prediction[val_mask], val_labels) == best_accuracy
    if history_path:
        with history_path.open('x', newline='') as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    result = {'best_val_accuracy': best_accuracy, 'selected_epoch': best_epoch + 1,
              'selected_epoch_zero_based': best_epoch, 'epochs_completed': epochs,
              'tie_rule': '>= selects later tied epoch', 'test_evaluated': False}
    return best_model, embedding, result

def guard_gpu(gpu_uuid):
    snapshot = gpu_snapshot()
    selected = next(gpu for gpu in snapshot['gpus'] if gpu['uuid'] == gpu_uuid)
    if selected['memory.free'] < 8192:
        raise RuntimeError('GPU has less than 8 GiB free; stopping without disturbing other processes')
    return snapshot, selected

def state_hash(model):
    digest_object = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest_object.update(name.encode())
        digest_object.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest_object.hexdigest()

def worker(batch, seed):
    preflight = json.loads((batch / 'preflight.json').read_text())
    assert preflight['status'] == 'passed'
    gpu = preflight['selected_gpu']
    directory = batch / ('seed_' + str(seed))
    directory.mkdir()
    status = {'status': 'running', 'seed': seed, 'pretrain_epochs_completed': 0, 'final_probe_epochs_completed': 0}
    status_file(directory / 'status.json', status)
    try:
        snapshot, current_gpu = guard_gpu(gpu['uuid'])
        if current_gpu['utilization.gpu'] >= 90:
            raise RuntimeError('Selected GPU is now heavily occupied; no training started')
        # UUID resolves nvidia-smi physical indices independently of CUDA ordering.
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu['uuid']
        os.environ['OMP_NUM_THREADS'] = '4'
        os.environ['MKL_NUM_THREADS'] = '4'
        start = time.perf_counter()
        import numpy as np
        import torch
        from sklearn.metrics import accuracy_score, f1_score
        sys.path.insert(0, str(REPO))
        import main_transductive as official_training
        from graphmae.models import build_model
        from graphmae.datasets.data_util import load_dataset
        from graphmae.utils import build_args, load_best_configs, create_optimizer
        torch.set_num_threads(4)
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        device = torch.device('cuda:0')
        assert torch.cuda.get_device_name(0) == gpu['name']
        assert execute(['git', '-C', REPO, 'rev-parse', 'HEAD']).strip() == COMMIT
        assert execute(['git', '-C', REPO, 'status', '--porcelain']).strip() == ''
        for name, value in preflight['data']['sha256'].items():
            assert digest(ROOT / 'data' / name) == value
        args_original = sys.argv
        sys.argv = ['main_transductive.py', '--dataset', 'cora', '--device', '0', '--seeds', str(seed), '--use_cfg']
        args = load_best_configs(build_args(), str(REPO / 'configs.yml'))
        sys.argv = args_original
        os.chdir(ROOT / 'data')
        graph, (features, classes) = load_dataset('cora')
        args.num_features = features
        for key, expected in EXPECTED.items():
            assert vars(args)[key] == expected
        assert classes == 7 and graph.num_nodes == 2708 and graph.num_edges == 13264
        graph = graph.to(device)
        # Test labels stay in a CPU-only sealed tensor; no probe receives them.
        full_labels = graph.y.cpu()
        train_mask, val_mask, test_mask = graph.train_mask, graph.val_mask, graph.test_mask
        train_labels = full_labels[train_mask.cpu()].to(device)
        val_labels = full_labels[val_mask.cpu()].to(device)
        del graph.y
        del graph.test_mask
        set_seed(seed)
        model = build_model(args).to(device)
        assert len(model.encoder.gat_layers) == 2 and len(model.decoder.gat_layers) == 1
        initial_token = model.enc_mask_token.detach().clone()
        optimizer = create_optimizer(args.optimizer, model, args.lr, args.weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (1 + np.cos(epoch * np.pi / args.max_epoch)) * 0.5)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        process_mapping = execute(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'])
        own_rows = [row for row in csv.reader(process_mapping.splitlines(), skipinitialspace=True) if len(row) >= 2 and row[1] == str(os.getpid())]
        assert own_rows and all(row[0] == gpu['uuid'] for row in own_rows), process_mapping
        metadata = {'seed': seed, 'official_args': vars(args), 'official_commit': COMMIT,
                    'preflight_batch': str(batch), 'data': preflight['data'],
                    'decoder_layers_verified': 1, 'gpu_before_start': snapshot,
                    'gpu_mapping': {'physical_nvidia_smi_index': gpu['physical_index'], 'physical_uuid': gpu['uuid'], 'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES'], 'CUDA_DEVICE_ORDER': os.environ['CUDA_DEVICE_ORDER'], 'logical_device': 'cuda:0', 'logical_device_count': 1, 'gpu_name': torch.cuda.get_device_name(0), 'own_process_mapping': own_rows, 'shared_at_start': current_gpu['memory.used'] > 512},
                    'rng': {'python_numpy_torch_cpu_cuda_seed': seed, 'cudnn_deterministic': torch.backends.cudnn.deterministic, 'cudnn_benchmark': torch.backends.cudnn.benchmark, 'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(), 'caveat': 'PyG scatter GPU atomics may remain nondeterministic'},
                    'protocol': {'periodic_probe_epochs': list(range(200, 1500, 200)), 'probe_epochs_each': 300, 'periodic_probe_RNG_isolation': False, 'final_encoder_epoch': 1500, 'classifier_selection': 'final probe validation >=, later ties', 'test_evaluations_planned': 1, 'test_labels_passed_to_training_or_probe': False},
                    'environment': preflight['environment'],
                    'runner_sources_sha256': {name: digest(ROOT / 'scripts' / name) for name in ['run_m1.py', 'preflight_m1.py']}}
        save_json(directory / 'effective_config.json', metadata)
        save_json(directory / 'model.json', {'repr': repr(model), 'parameter_count': sum(parameter.numel() for parameter in model.parameters())})
        torch.save({'train_mask': train_mask.cpu(), 'val_mask': val_mask.cpu(), 'test_mask': test_mask.cpu()}, directory / 'splits.pt')
        probes = directory / 'probes'
        probes.mkdir()
        periodic_results = []
        gpu_samples = []
        recorder = None
        def periodic_probe(model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob, mute=False):
            assert linear_prob
            epoch = len(recorder.rows)
            assert epoch in range(200, 1500, 200)
            gpu_state, _ = guard_gpu(gpu['uuid'])
            gpu_samples.append({'epoch': epoch, **gpu_state})
            encoder_before = state_hash(model.encoder)
            torch.cuda.synchronize()
            started = time.perf_counter()
            best_probe, _, result = validation_probe(model, graph, train_mask, val_mask, train_labels, val_labels, num_classes, lr_f, weight_decay_f, max_epoch_f, device, probes / f'epoch_{epoch:04d}_history.csv')
            torch.cuda.synchronize()
            result.update(pretrain_epoch=epoch, seconds=time.perf_counter() - started, encoder_unchanged=state_hash(model.encoder) == encoder_before)
            assert result['encoder_unchanged']
            save_json(probes / f'epoch_{epoch:04d}.json', result)
            periodic_results.append(result)
            print('Periodic validation probe:', result, flush=True)
            del best_probe
            return None, None  # official pretrain ignores returned accuracies
        official_training.node_classification_evaluation = periodic_probe
        class Recorder:
            def __init__(self, handle):
                self.rows = []
                self.writer = csv.DictWriter(handle, fieldnames=['epoch', 'epoch_zero_based', 'loss', 'lr_used', 'lr_next', 'training_cumulative_seconds'])
                self.writer.writeheader()
                self.handle = handle
                self.previous = time.perf_counter()
                self.training_seconds = 0
            def note(self, values, step):
                torch.cuda.synchronize()
                now = time.perf_counter()
                # Remove elapsed periodic-probe time from pretraining time.
                elapsed_probes = sum(result['seconds'] for result in periodic_results)
                self.training_seconds = now - pretrain_start - elapsed_probes
                row = {'epoch': step + 1, 'epoch_zero_based': step, 'loss': values['loss'],
                       'lr_used': args.lr * (1 + np.cos(step * np.pi / args.max_epoch)) * 0.5,
                       'lr_next': values['lr'], 'training_cumulative_seconds': self.training_seconds}
                if not np.isfinite(row['loss']):
                    raise RuntimeError('NaN/Inf in pretraining loss')
                if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
                    raise RuntimeError('NaN/Inf in pretraining gradients')
                if step == 0:
                    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.parameters())
                    assert not torch.equal(initial_token, model.enc_mask_token.detach())
                self.rows.append(row)
                self.writer.writerow(row)
                if (step + 1) % 50 == 0:
                    self.handle.flush()
                    status.update(pretrain_epochs_completed=step + 1)
                    status_file(directory / 'status.json', status)
        torch.cuda.synchronize()
        pretrain_start = time.perf_counter()
        with (directory / 'training.csv').open('x', newline='') as handle:
            recorder = Recorder(handle)
            model = official_training.pretrain(model, graph, graph.x, optimizer, args.max_epoch, device, scheduler, classes, args.lr_f, args.weight_decay_f, args.max_epoch_f, args.linear_prob, recorder)
        torch.cuda.synchronize()
        pretrain_with_periodic_seconds = time.perf_counter() - pretrain_start
        assert len(recorder.rows) == 1500
        assert [row['pretrain_epoch'] for row in periodic_results] == list(range(200, 1500, 200))
        assert all(row['epochs_completed'] == 300 and not row['test_evaluated'] for row in periodic_results)
        assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
        # Match the official CPU round-trip before its final linear evaluation.
        model = model.cpu()
        torch.save({'state_dict': model.state_dict(), 'official_args': vars(args), 'seed': seed, 'commit': COMMIT, 'epoch': 1500}, directory / 'pretrained_final.pt')
        model = model.to(device)
        model.eval()
        encoder_before = state_hash(model.encoder)
        torch.cuda.synchronize()
        final_probe_start = time.perf_counter()
        best_probe, embedding, final_selection = validation_probe(model, graph, train_mask, val_mask, train_labels, val_labels, classes, args.lr_f, args.weight_decay_f, 300, device, directory / 'final_probe_history.csv')
        torch.cuda.synchronize()
        final_probe_seconds = time.perf_counter() - final_probe_start
        assert state_hash(model.encoder) == encoder_before
        status.update(final_probe_epochs_completed=300)
        # Commit the selected classifier to disk before revealing test labels.
        save_json(directory / 'selection.json', final_selection)
        torch.save({'state_dict': {name: value.detach().cpu() for name, value in best_probe.state_dict().items()}, 'selection': final_selection, 'input_dim': 512, 'classes': classes, 'seed': seed, 'commit': COMMIT}, directory / 'classifier_best_val.pt')
        with torch.no_grad():
            logits = best_probe(graph, embedding)
            prediction = logits[test_mask].argmax(1).cpu().numpy()
        target = full_labels[test_mask.cpu()].numpy()
        test_accuracy = float(accuracy_score(target, prediction))
        test_micro_f1 = float(f1_score(target, prediction, average='micro'))
        assert abs(test_accuracy - test_micro_f1) < 1e-12
        test_nodes = torch.where(test_mask.cpu())[0].numpy()
        np.savez_compressed(directory / 'test_predictions.npz', node_id=test_nodes, target=target, prediction=prediction)
        torch.cuda.synchronize()
        result = {'status': 'passed', 'seed': seed, 'official_commit': COMMIT,
                  'pretrain_epochs_completed': 1500, 'periodic_probes_completed': 7,
                  'periodic_probe_epochs_each': 300, 'final_probe_epochs_completed': 300,
                  'best_val_accuracy': final_selection['best_val_accuracy'],
                  'selected_epoch': final_selection['selected_epoch'],
                  'selected_epoch_zero_based': final_selection['selected_epoch_zero_based'],
                  'selected_encoder_pretrain_epoch': 1500,
                  'test_accuracy': test_accuracy, 'test_micro_f1': test_micro_f1,
                  'test_evaluations': 1, 'test_used_for_selection': False,
                  'first_pretrain_loss': recorder.rows[0]['loss'], 'last_pretrain_loss': recorder.rows[-1]['loss'],
                  'pretrain_seconds_excluding_periodic_probes': pretrain_with_periodic_seconds - sum(row['seconds'] for row in periodic_results),
                  'periodic_probe_seconds': sum(row['seconds'] for row in periodic_results),
                  'final_probe_seconds': final_probe_seconds,
                  'training_and_probe_seconds': pretrain_with_periodic_seconds + final_probe_seconds,
                  'total_seconds': time.perf_counter() - start,
                  'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
                  'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
                  'gpu_mapping': metadata['gpu_mapping'], 'encoder_unchanged_by_probe': True,
                  'checkpoint_sha256': {name: digest(directory / name) for name in ['pretrained_final.pt', 'classifier_best_val.pt']}}
        save_json(directory / 'periodic_probes.json', periodic_results)
        save_json(directory / 'gpu_monitor.json', gpu_samples)
        save_json(directory / 'result.json', result)
        validate_result(directory)
        status_file(directory / 'status.json', {**status, 'status': 'passed', 'quality_check': 'passed'})
        print('FORMAL SEED RESULT', json.dumps(result, ensure_ascii=False), flush=True)
        return 0
    except Exception:
        error = traceback.format_exc()
        (directory / 'failure.txt').write_text(error)
        status_file(directory / 'status.json', {**status, 'status': 'failed', 'traceback': error})
        raise

def validate_result(directory):
    import numpy as np
    result = json.loads((directory / 'result.json').read_text())
    assert result['status'] == 'passed' and result['pretrain_epochs_completed'] == 1500
    assert result['final_probe_epochs_completed'] == 300 and result['periodic_probes_completed'] == 7
    assert result['test_evaluations'] == 1 and not result['test_used_for_selection']
    with (directory / 'training.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1500 and [int(row['epoch']) for row in rows] == list(range(1, 1501))
    assert all(np.isfinite(float(row['loss'])) for row in rows)
    with (directory / 'final_probe_history.csv').open() as handle:
        validation = list(csv.DictReader(handle))
    assert len(validation) == 300
    best_accuracy = max(float(row['val_accuracy']) for row in validation)
    selected = max(int(row['epoch']) for row in validation if float(row['val_accuracy']) == best_accuracy)
    assert result['best_val_accuracy'] == best_accuracy and result['selected_epoch'] == selected
    predicted = np.load(directory / 'test_predictions.npz')
    assert len(predicted['target']) == 1000
    assert float(np.mean(predicted['prediction'] == predicted['target'])) == result['test_accuracy']
    for name, value in result['checkpoint_sha256'].items():
        assert digest(directory / name) == value
    for name in ['effective_config.json', 'selection.json', 'splits.pt', 'periodic_probes.json', 'gpu_monitor.json']:
        assert (directory / name).is_file()
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, choices=range(5), default=[0])
    parser.add_argument('--worker-seed', type=int, choices=range(5))
    args = parser.parse_args()
    batch = args.batch.resolve()
    assert batch.is_relative_to(ROOT / 'results/raw/M1/cora')
    if args.worker_seed is not None:
        return worker(batch, args.worker_seed)
    assert len(set(args.seeds)) == len(args.seeds) and args.seeds == sorted(args.seeds)
    # Seed zero must pass the complete artifact gate before starting later seeds.
    if args.seeds[0] != 0:
        validate_result(batch / 'seed_0')
    preflight = json.loads((batch / 'preflight.json').read_text())
    assert preflight['status'] == 'passed'
    for seed in args.seeds:
        if (batch / f'seed_{seed}').exists():
            raise RuntimeError('Existing seed directory; use a fresh preflight batch for reruns')
        log = batch / f'seed_{seed}_console.log'
        with log.open('x') as output:
            process = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--batch', str(batch), '--worker-seed', str(seed)], stdout=output, stderr=subprocess.STDOUT)
        if process.returncode:
            print(log.read_text()[-9000:], flush=True)
            raise SystemExit('Seed failed; remaining runs stopped. Evidence: ' + str(log))
        result = validate_result(batch / f'seed_{seed}')
        print(json.dumps({'seed': seed, 'quality_check': 'passed', 'val_accuracy': result['best_val_accuracy'], 'test_accuracy': result['test_accuracy'], 'total_seconds': result['total_seconds'], 'result_directory': str(batch / f'seed_{seed}')}, ensure_ascii=False), flush=True)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
