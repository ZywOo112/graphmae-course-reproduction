import concurrent.futures
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import urllib.request

ROOT = Path('/home/fuqiang/homework/机器学习课程大作业/GraphMAE')
ROOT.mkdir(parents=True, exist_ok=True)
for name in ['docs/paper', 'third_party', 'configs', 'scripts', 'results/raw/M0', 'results/processed', 'figures', 'reports/figures', 'reports/appendices', 'agent_logs/M0']:
    (ROOT / name).mkdir(parents=True, exist_ok=True)

def save(name, text):
    path = ROOT / name
    with path.open('x') as f:
        f.write(text)

def run(command, name, timeout=180):
    p = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    save('results/raw/M0/' + name + '.txt', '$ ' + ' '.join(command) + '\n' + p.stdout + '\nexit_code=' + str(p.returncode) + '\n')
    print(name, p.returncode, p.stdout[:9000], flush=True)
    return p

# Select only the real user task from this session, never environment/system text.
session = Path('/home/fuqiang/.codex/sessions/2026/11/12/rollout-2026-11-12T03-48-51-01a1cddf-6163-7061-a1f3-bdf5774aa614.jsonl')
matches = []
for line in session.read_text().splitlines():
    data = json.loads(line)
    payload = data.get('payload', {})
    if data.get('type') == 'response_item' and payload.get('role') == 'user':
        content = ''.join(c.get('text', '') for c in payload.get('content', []) if c.get('type') == 'input_text')
        if content.startswith('# GraphMAE 课程大作业：M0 项目初始化与可行性审计'):
            matches.append(content)
assert len(matches) == 1, len(matches)
save('agent_logs/M0/prompt.md', matches[0])
save('agent_logs/M0/provenance.json', json.dumps({'session_id': '01a1cddf-6163-7061-a1f3-bdf5774aa614', 'prompt_source': 'response_item/user/input_text', 'prompt_sha256': hashlib.sha256(matches[0].encode()).hexdigest(), 'response_scope': 'assistant final response; exact UTF-8 text; commentary separately exported when session is finalized', 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}, ensure_ascii=False, indent=2) + '\n')

commands = [(['bash', '-lc', 'cat /etc/os-release; lscpu; free -h; df -h /home/fuqiang /tmp; nvidia-smi; command -v python python3 conda uv nvcc xelatex latexmk pdflatex tectonic; python3 --version; conda --version; uv --version; ls -d /usr/local/cuda* /opt/texlive* /usr/local/texlive* 2>/dev/null; fc-list :lang=zh family | head -15'], 'server'),
(['python3', '-c', "import importlib.util,importlib.metadata as m,json; print(json.dumps({p: (m.version(p) if importlib.util.find_spec(s) else None) for p,s in [('torch','torch'),('torch-geometric','torch_geometric'),('dgl','dgl'),('numpy','numpy')]},indent=2)); import torch; print('CUDA build',torch.version.cuda,'available',torch.cuda.is_available())"], 'base_python'),
(['conda', 'env', 'list', '--json'], 'conda_envs')]
with concurrent.futures.ThreadPoolExecutor() as pool:
    list(pool.map(lambda item: run(*item), commands))

repo = ROOT / 'third_party/GraphMAE'
if not repo.exists():
    run(['git', 'clone', '--branch', 'pyg', 'https://github.com/THUDM/GraphMAE.git', str(repo)], 'clone', 240)
run(['git', '-C', str(repo), 'show-ref'], 'git_refs')
run(['git', '-C', str(repo), 'status', '--short'], 'git_status_initial')
run(['git', '-C', str(repo), 'ls-tree', '-r', 'HEAD'], 'pyg_tree')
run(['git', '-C', str(repo), 'ls-tree', '-r', 'origin/main'], 'main_tree')
main_repo = ROOT / 'third_party/GraphMAE-main-audit'
if not main_repo.exists():
    run(['git', '-C', str(repo), 'worktree', 'add', '--detach', str(main_repo), 'origin/main'], 'main_worktree')

def download(url, path):
    with urllib.request.urlopen(url, timeout=90) as response:
        data = response.read()
    with (ROOT / path).open('xb') as f:
        f.write(data)
    print('download', url, len(data), flush=True)
    return hashlib.sha256(data).hexdigest()

paper_hash = download('https://arxiv.org/pdf/2205.10803v3', 'docs/paper/GraphMAE.pdf')
save('docs/paper/provenance.json', json.dumps({'url': 'https://arxiv.org/pdf/2205.10803v3', 'sha256': paper_hash, 'accessed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}, indent=2) + '\n')
run(['pdftotext', '-layout', str(ROOT / 'docs/paper/GraphMAE.pdf'), str(ROOT / 'docs/paper/GraphMAE.txt')], 'paper_extract')
print('BOOTSTRAP COMPLETE', flush=True)
