"""Resume an unstarted final test after a GPU guard failure; never retrain."""
import argparse
import csv
import datetime
import os
import sys
from pathlib import Path
import time

import run_m2
from run_m2 import REPO, arm, assert_inputs, save_json
from preflight_m1 import gpu_snapshot, digest, execute

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--degree-alpha', type=float, required=True)
    parser.add_argument('--physical-gpu', type=int, required=True)
    cli = parser.parse_args()
    batch = cli.batch.resolve()
    preflight = assert_inputs(batch)
    directory = batch / arm(cli.degree_alpha) / f'seed_{cli.seed}'
    assert not any((directory / name).exists() for name in ['test_started.json', 'test_predictions.npz', 'final_test.json'])
    old_log = directory / 'test.log'
    assert 'GPU has less than 8 GiB free' in old_log.read_text()
    snapshots = []
    for index in range(3):
        if index:
            time.sleep(2)
        snapshots.append(gpu_snapshot())
    devices = [next(g for g in snapshot['gpus'] if g['index'] == cli.physical_gpu) for snapshot in snapshots]
    assert min(g['memory.free'] for g in devices) >= 8192
    assert max(g['utilization.gpu'] for g in devices) < 90
    selected = devices[-1]
    gpu = {'physical_index': selected['index'], 'uuid': selected['uuid'], 'name': selected['name'], 'shared': selected['memory.used'] > 512}
    context = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'reason': 'original GPU guard failed before test_started marker or model forward; resume only untested model',
               'degree_alpha': cli.degree_alpha, 'seed': cli.seed, 'training_gpu': preflight['selected_gpu'],
               'test_gpu': gpu, 'logical_device': 'cuda:0', 'gpu_snapshots': snapshots,
               'original_failure_log_sha256': digest(old_log), 'alpha_selection_sha256': digest(batch / 'alpha_selection.json'),
               'core_runner_unchanged': True, 'no_training_or_alpha_reselection': True,
               'limitation': 'This one final inference uses A40 rather than L40; floating-point reduction differences remain possible.'}
    context_path = directory / 'test_device_override.json'
    attempt = 2
    while context_path.exists():
        context_path = directory / f'test_device_override_attempt{attempt}.json'
        attempt += 1
    save_json(context_path, context)
    original_setup = run_m2.setup_gpu
    def replacement_setup(current_batch):
        assert current_batch == batch
        snapshot, current = run_m2.guard_gpu(gpu['uuid'])
        assert current['utilization.gpu'] < 90
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu['uuid']
        os.environ['OMP_NUM_THREADS'] = '4'
        os.environ['MKL_NUM_THREADS'] = '4'
        return preflight, gpu, snapshot, current
    run_m2.setup_gpu = replacement_setup
    sys.path.insert(0, str(REPO))
    try:
        run_m2.test_worker(batch, cli.degree_alpha, cli.seed)
    finally:
        run_m2.setup_gpu = original_setup
    process_info = execute(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'])
    own = [row for row in csv.reader(process_info.splitlines(), skipinitialspace=True) if len(row) >= 2 and row[1] == str(os.getpid())]
    assert own and all(row[0] == gpu['uuid'] for row in own)
    assert digest(batch / 'alpha_selection.json') == context['alpha_selection_sha256']
    save_json(directory / 'test_resume_verification.json', {'status': 'passed', 'physical_gpu': gpu, 'own_process_mapping': own, 'context_file': context_path.name,
                                                          'test_evaluations': 1, 'original_failed_launch_did_not_evaluate': True})

if __name__ == '__main__':
    main()
