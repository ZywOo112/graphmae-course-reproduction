"""Final M1 artifact review; no training, model selection or test forward pass."""
import csv
import datetime
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

from run_m1 import ROOT, REPO, COMMIT, validate_result, status_file
from preflight_m1 import digest, execute

def main():
    import numpy as np
    summary = json.loads((ROOT / 'results/processed/M1/summary.json').read_text())
    batch = ROOT / summary['batch']
    results = [validate_result(batch / f'seed_{seed}') for seed in range(5)]
    accuracy = np.array([result['test_accuracy'] for result in results])
    assert float(accuracy.mean()) == summary['accuracy_mean']
    assert float(accuracy.std(ddof=1)) == summary['accuracy_sample_std']
    assert all(result['test_micro_f1'] == result['test_accuracy'] for result in results)
    assert execute(['git', '-C', REPO, 'rev-parse', 'HEAD']).strip() == COMMIT
    assert not execute(['git', '-C', REPO, 'status', '--porcelain']).strip()
    preflight = json.loads((batch / 'preflight.json').read_text())
    for name, value in preflight['data']['sha256'].items():
        assert digest(ROOT / 'data' / name) == value
    for name, value in preflight['official_files_sha256'].items():
        assert digest(REPO / name) == value
    for path in sorted((ROOT / 'scripts').glob('*.py')):
        compile(path.read_text(), str(path), 'exec')
    plot = json.loads((ROOT / 'figures/M1/plot_metadata.json').read_text())
    image_details = []
    assert len(plot['figures']) == 5 and plot['dpi'] == 300
    for record in plot['figures']:
        path = ROOT / record['file']
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == record['sha256']
        assert data[:8] == b'\x89PNG\r\n\x1a\n'
        width, height = struct.unpack('>II', data[16:24])
        assert width >= 2000 and height >= 1200
        image_details.append({'file': record['file'], 'width': width, 'height': height, 'sha256': record['sha256']})
    required = ['README.md', 'PROJECT_PLAN.md', 'PROGRESS.md', 'DECISIONS.md', 'SKILL.md', 'AGENT_WORKFLOW.md', 'M1_ACCEPTANCE.md']
    assert all((ROOT / name).is_file() for name in required)
    execute([sys.executable, '/home/fuqiang/.codex/skills/.system/skill-creator/scripts/quick_validate.py', ROOT])
    execute([sys.executable, ROOT / 'scripts/generate_appendices.py'])
    manifest = json.loads((ROOT / 'reports/appendices/manifest.json').read_text())
    assert all(path.relative_to(ROOT).as_posix() in manifest['sources'] for path in (ROOT / 'scripts').glob('*.py'))
    for name, value in manifest['sha256'].items():
        assert digest(ROOT / name) == value
    final = {'status': 'M1_complete_paused', 'seeds_completed': list(range(5)), 'all_quality_checks_passed': True, 'training_started': True, 'M2_started': False, 'acceptance': 'M1_ACCEPTANCE.md'}
    status_file(batch / 'status.json', final)
    status_file(batch / 'final_status.json', final)
    artifacts = []
    for base in [batch, ROOT / 'results/processed/M1', ROOT / 'figures/M1']:
        for path in sorted(base.rglob('*')):
            if path.is_file():
                artifacts.append({'file': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size, 'sha256': digest(path)})
    report = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'status': 'passed', 'seeds_completed': [0, 1, 2, 3, 4], 'per_seed_pretraining_epochs': 1500, 'per_seed_periodic_probe_count': 7, 'per_probe_epochs': 300, 'final_probe_epochs': 300, 'accuracy_mean': summary['accuracy_mean'], 'accuracy_sample_std_ddof1': summary['accuracy_sample_std'], 'all_result_files_and_metrics_checked': True, 'all_python_source_syntax': 'passed', 'official_source_and_M0_data_sha256': 'unchanged', 'project_skill_validation': 'passed', 'appendix_complete_source_inclusion': True, 'pngs': image_details, 'latex_compilation_attempted': False, 'agent_listener_started': False, 'M2_started': False}
    output = ROOT / 'results/processed/M1/final_verification.json'
    with output.open('x') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    artifacts.append({'file': str(output.relative_to(ROOT)), 'bytes': output.stat().st_size, 'sha256': digest(output)})
    with (ROOT / 'results/processed/M1/artifact_manifest.json').open('x') as handle:
        json.dump({'batch': summary['batch'], 'artifacts': artifacts}, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
