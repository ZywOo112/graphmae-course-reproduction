"""Create the project's isolated environment; never install into base Python."""
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / '.venv'

def run(command, log):
    print('Running:', command, flush=True)
    with (ROOT / 'results/raw/M0' / log).open('x') as output:
        process = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)
    if process.returncode:
        print((ROOT / 'results/raw/M0' / log).read_text()[-6000:], flush=True)
        raise SystemExit(process.returncode)

if ENV.exists():
    raise SystemExit('Environment already exists; inspect it instead of overwriting.')
run([sys.executable, '-m', 'venv', str(ENV)], 'venv_create.txt')
python = str(ENV / 'bin/python')
run(['uv', 'pip', 'install', '--python', python, 'torch==2.4.1', '--index-url', 'https://download.pytorch.org/whl/cu121'], 'install_torch.txt')
run(['uv', 'pip', 'install', '--python', python, '--only-binary', ':all:', '--no-deps', 'torch-scatter==2.1.2', 'torch-sparse==0.6.18', '--find-links', 'https://data.pyg.org/whl/torch-2.4.0+cu121.html'], 'install_extensions.txt')
run(['uv', 'pip', 'install', '--python', python, '-r', str(ROOT / 'configs/requirements.in')], 'install_dependencies.txt')
run(['uv', 'pip', 'check', '--python', python], 'dependency_check.txt')
run(['uv', 'pip', 'freeze', '--python', python], 'freeze.txt')
# uv's environment banner is stderr, not a requirement specification.
lock = subprocess.check_output(['uv', 'pip', 'freeze', '--python', python], text=True)
with (ROOT / 'configs/requirements.lock.txt').open('x') as output:
    output.write('# Python 3.11; torch cu121; extensions from data.pyg.org\n' + lock)
print('Environment ready:', python, flush=True)
