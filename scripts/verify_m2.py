"""M2 delivery review without training, model inference or alpha reselection."""
import csv
import datetime
import json
import struct
import sys

from run_m2 import ROOT, ALPHAS, arm, assert_inputs, validate_training, check_paired_rng, save_json, status_file
from preflight_m1 import digest, execute

def main():
    import numpy as np
    summary = json.loads((ROOT / 'results/processed/M2/summary.json').read_text())
    batch = ROOT / summary['batch']
    preflight = assert_inputs(batch)
    paired_rng = check_paired_rng(batch, False)
    pilot = json.loads((batch / 'pilot_check.json').read_text())
    assert pilot['status'] == 'passed' and pilot['test_evaluations'] == 0
    assert json.loads((batch / 'masking_checks.json').read_text())['status'] == 'passed'
    assert all(validate_training(batch / 'pilot' / arm(alpha) / 'seed_0', True)['test_evaluations'] == 0 for alpha in ALPHAS)
    selection = json.loads((batch / 'alpha_selection.json').read_text())
    selected = selection['selected_degree_alpha']
    assert selected == summary['selected_degree_alpha']
    assert not selection['test_results_available_at_selection']
    assert summary['selection_sha256'] == digest(batch / 'alpha_selection.json')
    formal = []
    tests = []
    for alpha in ALPHAS:
        for seed in range(5):
            directory = batch / arm(alpha) / f'seed_{seed}'
            formal.append(validate_training(directory))
            metadata = json.loads((directory / 'effective_config.json').read_text())
            assert metadata['official_args']['alpha_l'] == 3 and metadata['degree_alpha'] == alpha
            assert metadata['initial_model_sha256'] == formal[-1]['initial_model_sha256']
            assert metadata['gpu']['uuid'] == preflight['selected_gpu']['uuid']
            for name, value in metadata['sources_sha256'].items():
                assert digest(ROOT / 'scripts' / name) == value
            for name, value in selection['inputs_sha256'][str(directory.relative_to(batch))].items():
                assert digest(directory / name) == value
            if alpha in [0.0, selected]:
                test = json.loads((directory / 'final_test.json').read_text())
                assert test['test_evaluations'] == 1 and test['selection_sha256'] == summary['selection_sha256']
                assert (directory / 'test_started.json').stat().st_mtime_ns > (batch / 'alpha_selection.json').stat().st_mtime_ns
                predicted = np.load(directory / 'test_predictions.npz')
                accuracy = float(np.mean(predicted['prediction'] == predicted['target']))
                assert accuracy == test['test_accuracy'] == test['test_micro_f1']
                tests.append(test)
            else:
                assert not any((directory / name).exists() for name in ['test_started.json', 'final_test.json', 'test_predictions.npz'])
    assert len(formal) == 20 and len(tests) == 10
    device_exception = json.loads((batch / 'degree_alpha_0.25/seed_4/test_resume_verification.json').read_text())
    assert device_exception['status'] == 'passed' and device_exception['test_evaluations'] == 1
    assert device_exception['original_failed_launch_did_not_evaluate']
    differences = np.array([r['difference_percentage_points'] for r in summary['paired_records']])
    assert float(differences.mean()) == summary['paired_difference_mean_percentage_points']
    assert float(differences.std(ddof=1)) == summary['paired_difference_sample_std_percentage_points']
    for path, value in json.loads((batch / 'previous_milestones_sha256.json').read_text()).items():
        assert digest(ROOT / path) == value
    plot = json.loads((ROOT / 'figures/M2/plot_metadata.json').read_text())
    assert plot['dpi'] == 300 and len(plot['figures']) == 5
    assert plot['summary_sha256'] == digest(ROOT / 'results/processed/M2/summary.json')
    pngs = []
    for item in plot['figures']:
        path = ROOT / item['file']
        assert digest(path) == item['sha256']
        data = path.read_bytes()
        assert data[:8] == b'\x89PNG\r\n\x1a\n'
        width, height = struct.unpack('>II', data[16:24])
        assert width >= 2000 and height >= 1200
        pngs.append({'file': item['file'], 'width': width, 'height': height})
    for path in (ROOT / 'scripts').glob('*.py'):
        compile(path.read_text(), str(path), 'exec')
    for name in ['M2_ACCEPTANCE.md', 'PROGRESS.md', 'DECISIONS.md', 'PROJECT_PLAN.md', 'README.md', 'SKILL.md', 'AGENT_WORKFLOW.md']:
        assert (ROOT / name).is_file()
    execute([sys.executable, '/home/fuqiang/.codex/skills/.system/skill-creator/scripts/quick_validate.py', ROOT])
    execute([sys.executable, ROOT / 'scripts/generate_appendices.py'])
    appendix = json.loads((ROOT / 'reports/appendices/manifest.json').read_text())
    assert all(path.relative_to(ROOT).as_posix() in appendix['sources'] for path in (ROOT / 'scripts').glob('*.py'))
    final_status = {'status': 'M2_complete_paused', 'formal_training_runs': 20, 'pilot_runs': 4, 'selected_degree_alpha': selected,
                    'final_tests': 10, 'M3_started': False, 'acceptance': 'M2_ACCEPTANCE.md'}
    status_file(batch / 'status.json', final_status)
    report = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), **final_status,
              'test_selection_order_verified': True, 'no_unselected_coefficient_tested': True,
              'initial_weights_and_all_1500_epoch_rng_traces_equal_per_seed': paired_rng,
              'M0_M1_and_official_source_preserved': True, 'mask_budget_per_epoch': 1354,
              'plots': pngs, 'source_syntax_and_skill': 'passed', 'appendix_sources': len(appendix['sources']),
              'agent_listener_started': False, 'latex_compilation_attempted': False}
    report['final_inference_device_exception'] = device_exception
    output = ROOT / 'results/processed/M2/final_verification.json'
    save_json(output, report)
    artifacts = {}
    for directory in [batch, ROOT / 'results/processed/M2', ROOT / 'figures/M2']:
        for path in directory.rglob('*'):
            if path.is_file():
                artifacts[str(path.relative_to(ROOT))] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
    save_json(ROOT / 'results/processed/M2/artifact_manifest.json', {'batch': summary['batch'], 'artifacts': artifacts})
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)

if __name__ == '__main__':
    main()
