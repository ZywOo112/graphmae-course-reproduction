"""Review M0 deliverables without launching further model training."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / '.venv/bin/python'
SESSION = Path('/home/fuqiang/.codex/sessions/2026/11/12/rollout-2026-11-12T03-48-51-01a1cddf-6163-7061-a1f3-bdf5774aa614.jsonl')

def run(command, expected=0):
    result = subprocess.run(list(map(str, command)), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
    print('Command:', command, '\n', result.stdout, 'exit_code=', result.returncode, flush=True)
    assert result.returncode == expected, result.stdout
    return result.stdout

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch-logs', action='store_true')
    args = parser.parse_args()
    raw = ROOT / 'results/raw/M0'
    required = ['README.md', 'PROJECT_PLAN.md', 'PROGRESS.md', 'DECISIONS.md', 'SKILL.md', 'AGENT_WORKFLOW.md', 'docs/SOURCE_AUDIT.md', 'docs/ENVIRONMENT_AUDIT.md', 'docs/paper/PAPER_NOTES.md', 'docs/paper/GraphMAE.pdf', 'reports/main.tex', 'reports/references.bib', 'agent_logs/M0/prompt.md', 'third_party/GraphMAE.bundle']
    assert all((ROOT / name).is_file() for name in required)
    for path in sorted((ROOT / 'scripts').glob('*.py')):
        compile(path.read_text(), str(path), 'exec')
    lock = ROOT / 'configs/requirements.lock.txt'
    code = 'from packaging.requirements import Requirement; from pathlib import Path; p=Path(' + repr(str(lock)) + '); lines=[x for x in p.read_text().splitlines() if x and not x.startswith("#")]; [Requirement(x) for x in lines]; print("Valid pinned requirements:", len(lines))'
    lock_check = run([PYTHON, '-c', code])
    dependency_check = run(['uv', 'pip', 'check', '--python', PYTHON])
    lock_dry_run = run(['uv', 'pip', 'install', '--offline', '--dry-run', '--python', PYTHON, '-r', lock])
    skill_validation = run([sys.executable, '/home/fuqiang/.codex/skills/.system/skill-creator/scripts/quick_validate.py', ROOT])
    paper = json.loads((ROOT / 'docs/paper/provenance.json').read_text())
    assert hashlib.sha256((ROOT / 'docs/paper/GraphMAE.pdf').read_bytes()).hexdigest() == paper['sha256']
    smoke_path = raw / 'smoke_20261112T040413960326Z/summary.json'
    smoke = json.loads(smoke_path.read_text())
    assert smoke['status'] == 'passed' and not smoke['is_formal_experiment']
    assert smoke['pretrain_epochs'] == 20 and smoke['probe_epochs'] == 20
    for name, digest in smoke['data_sha256'].items():
        assert hashlib.sha256((ROOT / 'data' / name).read_bytes()).hexdigest() == digest
    repositories = {}
    for name, commit in [('GraphMAE', '9127f25e01aa5ba1cca68b746284aeb7ab2b4850'), ('GraphMAE-main-audit', 'b14f080c919257b495e3cb6474286d5252d6a635')]:
        path = ROOT / 'third_party' / name
        actual = run(['git', '-C', path, 'rev-parse', 'HEAD']).strip()
        status = run(['git', '-C', path, 'status', '--porcelain']).strip()
        assert actual == commit and status == ''
        repositories[name] = {'commit': actual, 'status': status}
    # The real final response cannot exist until the response has been sent.
    export_status = subprocess.run([str(PYTHON), str(ROOT / 'scripts/export_agent_logs.py'), '--session', str(SESSION), '--stage', 'M0'], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert export_status.returncode in {0, 3}, export_status.stdout
    print(export_status.stdout, flush=True)
    build_output = run([PYTHON, ROOT / 'scripts/build_report.py'], expected=2)
    manifest = json.loads((ROOT / 'reports/appendices/manifest.json').read_text())
    expected_sources = {p.relative_to(ROOT).as_posix() for p in (ROOT / 'scripts').glob('*.py')}
    assert expected_sources.issubset(set(manifest['sources']))
    appendix = (ROOT / 'reports/appendices/generated.tex').read_text()
    imports = re.findall(r'\\appendix(?:file|text)\{([^}]+)\}', appendix)
    assert all((ROOT / 'reports' / name).resolve().is_file() for name in imports)
    for name, digest in manifest['sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    disk = run(['du', '-s', '-B1', ROOT / '.venv', ROOT / 'data', ROOT / 'third_party', ROOT / 'results', ROOT / 'reports'])
    # A complete project file list, excluding reconstructible packages and bulk data.
    file_list = []
    for directory, dirs, files in os.walk(ROOT):
        dirs[:] = sorted(d for d in dirs if d not in {'.venv', '.git', '__pycache__'})
        for name in sorted(files):
            file_list.append((Path(directory) / name).relative_to(ROOT).as_posix())
    (ROOT / 'docs/PROJECT_FILES.txt').write_text('\n'.join(file_list) + '\n')
    status = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'required_files_exist': True, 'python_source_syntax': 'passed', 'requirements_lock_validation': lock_check, 'dependency_check': dependency_check, 'lock_offline_dry_run': lock_dry_run, 'skill_validation': skill_validation, 'data_and_paper_sha256': 'passed', 'repositories': repositories, 'smoke_summary': smoke_path.relative_to(ROOT).as_posix(), 'smoke_was_not_rerun': True, 'appendix_import_count': len(imports), 'self_written_source_and_configs_count': len(manifest['sources']), 'all_annex_paths_and_hashes_checked': True, 'latex': json.loads((ROOT / 'reports/build_status.json').read_text()), 'agent_final_export': 'waiting_for_actual_final_message' if export_status.returncode == 3 else 'exported', 'disk_usage_bytes': disk, 'formal_M1_started': False}
    (raw / 'final_check.json').write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
    if args.watch_logs:
        with (raw / 'agent_log_watcher.log').open('a') as log:
            process = subprocess.Popen([str(PYTHON), str(ROOT / 'scripts/export_agent_logs.py'), '--session', str(SESSION), '--stage', 'M0', '--watch', '--timeout', '1800'], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        (raw / 'agent_log_watcher.json').write_text(json.dumps({'pid': process.pid, 'session': str(SESSION), 'purpose': 'export actual final response after it is sent; never run experiments'}, indent=2) + '\n')
        print('Started actual-response archive watcher:', process.pid, flush=True)
    print('M0 review passed; LaTeX compilation remains blocked by missing tools. No M1 training executed.', flush=True)

if __name__ == '__main__':
    main()
