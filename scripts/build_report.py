"""Build the XeLaTeX report when available, otherwise record a real blocker."""
import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports'

def main():
    subprocess.run([sys.executable, str(ROOT / 'scripts/generate_appendices.py')], check=True)
    status = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'tools': {name: shutil.which(name) for name in ['xelatex', 'latexmk', 'bibtex']}, 'installed_system_packages': False}
    if not status['tools']['xelatex'] or not status['tools']['bibtex']:
        status.update(status='blocked_missing_tex', compiled=False, pdf_created=False, missing=[k for k, v in status['tools'].items() if not v], next_action='Use a machine with TeX Live (XeLaTeX, ctex/Fandol, fvextra, natbib and BibTeX); compile the full project directory. No system installation performed.')
        (REPORTS / 'build_status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 2
    if status['tools']['latexmk']:
        commands = [['latexmk', '-xelatex', '-interaction=nonstopmode', '-halt-on-error', 'main.tex']]
    else:
        commands = [['xelatex', '-interaction=nonstopmode', '-halt-on-error', 'main.tex'], ['bibtex', 'main'], ['xelatex', '-interaction=nonstopmode', '-halt-on-error', 'main.tex'], ['xelatex', '-interaction=nonstopmode', '-halt-on-error', 'main.tex']]
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    returncode = 0
    with (REPORTS / ('build_' + stamp + '.log')).open('x') as log:
        for command in commands:
            process = subprocess.run(command, cwd=REPORTS, stdout=log, stderr=subprocess.STDOUT)
            if process.returncode:
                returncode = process.returncode
                break
    status.update(status='passed' if returncode == 0 else 'failed', compiled=returncode == 0, pdf_created=(REPORTS / 'main.pdf').exists(), returncode=returncode, commands=commands)
    (REPORTS / 'build_status.json').write_text(json.dumps(status, indent=2) + '\n')
    return returncode

if __name__ == '__main__':
    raise SystemExit(main())
