"""Collect immutable validation evidence and run only the bounded M0 test."""
from pathlib import Path
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'results/raw/M0' / ('validation_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
PYTHON = ROOT / '.venv/bin/python'

def command(args, name, timeout=180):
    path = RAW / (name + '.txt')
    with path.open('x') as output:
        output.write('$ ' + ' '.join(map(str, args)) + '\n')
        output.flush()
        result = subprocess.run(list(map(str, args)), stdout=output, stderr=subprocess.STDOUT, timeout=timeout)
        output.write('\nexit_code=' + str(result.returncode) + '\n')
    print(name, result.returncode, path.read_text()[-10000:], flush=True)
    return result.returncode

def package_audit():
    packages = ['torch', 'torch-geometric', 'dgl', 'numpy', 'torch-scatter', 'torch-sparse']
    interpreters = [Path('/home/fuqiang/anaconda3/bin/python'), PYTHON]
    interpreters += sorted(Path('/home/fuqiang/anaconda3/envs').glob('*/bin/python'))
    code = "import importlib.metadata as m,json,sys; out={'python':sys.version};\nfor name in " + repr(packages) + ":\n try: out[name]=m.version(name)\n except m.PackageNotFoundError: out[name]=None\nprint(json.dumps(out))"
    def inspect(python):
        result = subprocess.run([str(python), '-c', code], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        return str(python), {'returncode': result.returncode, 'output': result.stdout, 'stderr': result.stderr}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        return dict(pool.map(inspect, interpreters))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--allow-shared', action='store_true')
    args = parser.parse_args()
    RAW.mkdir()
    # Preserve the actual bootstrap source as part of the self-written code annex.
    bootstrap_copy = ROOT / 'scripts/bootstrap_m0.py'
    if not bootstrap_copy.exists():
        shutil.copyfile('/tmp/graphmae_m0_bootstrap.py', bootstrap_copy)
    audit = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'python_environments_read_only': package_audit(), 'tools': {name: shutil.which(name) for name in ['uv', 'conda', 'xelatex', 'latexmk', 'bibtex', 'pdflatex', 'tectonic', 'pdftotext', 'fc-list']}}
    audit['cuda_toolkit_nvcc'] = '/usr/local/cuda/bin/nvcc' if Path('/usr/local/cuda/bin/nvcc').exists() else None
    (RAW / 'environment_details.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    command(['/usr/local/cuda/bin/nvcc', '--version'], 'nvcc')
    command(['nvidia-smi', '--query-gpu=index,name,uuid,memory.total,memory.used,driver_version', '--format=csv'], 'gpu_before_smoke')
    command(['fc-list', ':lang=zh', 'family'], 'chinese_fonts') if shutil.which('fc-list') else None
    command([PYTHON, '-c', "from pypdf import PdfReader; from pathlib import Path; p=Path(" + repr(str(ROOT / 'docs/paper/GraphMAE.pdf')) + "); r=PdfReader(p); p.with_suffix('.txt').write_text('\\n\\n'.join(page.extract_text() for page in r.pages)); print('pages',len(r.pages))"], 'paper_extract_pypdf')
    smoke_command = [PYTHON, ROOT / 'scripts/smoke_test.py', '--gpu', str(args.gpu)]
    if args.allow_shared:
        smoke_command.append('--allow-shared')
    code = command(smoke_command, 'smoke_console', timeout=600)
    if code:
        raise SystemExit(code)
    repo = ROOT / 'third_party/GraphMAE'
    # A portable full-history bundle allows restoration without network/worktree metadata.
    bundle = ROOT / 'third_party/GraphMAE.bundle'
    if not bundle.exists():
        command(['git', '-C', repo, 'bundle', 'create', bundle, '--all'], 'git_bundle')
    command(['git', '-C', repo, 'bundle', 'verify', bundle], 'git_bundle_verify')
    verification = {}
    for name in ['GraphMAE', 'GraphMAE-main-audit']:
        directory = ROOT / 'third_party' / name
        verification[name] = {key: subprocess.check_output(['git', '-C', str(directory)] + args, text=True).strip() for key, args in {'commit': ['rev-parse', 'HEAD'], 'status_porcelain': ['status', '--porcelain'], 'remote': ['remote', 'get-url', 'origin']}.items()}
    verification['bundle_sha256'] = hashlib.sha256(bundle.read_bytes()).hexdigest()
    (RAW / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    with (ROOT / 'results/raw/M0/verification.json').open('x') as output:
        output.write(json.dumps(verification, indent=2) + '\n')
    print('M0 VALIDATION COMPLETE', flush=True)

if __name__ == '__main__':
    main()
