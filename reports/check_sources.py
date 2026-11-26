"""Static source checks only: never invokes TeX, models, or appendix generators.

Run from project root: .venv/bin/python reports/check_sources.py
Writes only reports/M3_2_STATIC_CHECK.json. Not a LaTeX parser/compiler.
"""
from pathlib import Path
import argparse
import collections
import datetime
import hashlib
import json
import re
import struct
import subprocess

REPORTS = Path(__file__).resolve().parent
ROOT = REPORTS.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def uncomment(text):
    return re.sub(r'(?<!\\)%[^\n]*', '', text)


def raster_info(path):
    data = path.read_bytes()
    if data.startswith(b'\x89PNG\r\n\x1a\n'):
        return {'format': 'PNG', 'pixels': list(struct.unpack('>II', data[16:24]))}
    if data.startswith(b'\xff\xd8'):
        offset = 2
        while offset < len(data):
            if data[offset] != 255:
                break
            while data[offset] == 255:
                offset += 1
            marker = data[offset]
            offset += 1
            if marker in [0xD8, 0xD9] or 0xD0 <= marker <= 0xD7:
                continue
            length = int.from_bytes(data[offset:offset+2], 'big')
            if marker in [0xC0, 0xC1, 0xC2]:
                height, width = struct.unpack('>HH', data[offset+3:offset+7])
                return {'format': 'JPEG', 'pixels': [width, height]}
            offset += length
        return {'format': 'JPEG', 'pixels': None}
    return {'format': 'unknown', 'pixels': None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='M3_2_STATIC_CHECK.json')
    output_name = parser.parse_args().output
    errors = []
    inputs, graphics, sources = [], [], {}

    def inspect(path):
        relative = path.relative_to(ROOT).as_posix()
        if relative in sources:
            return
        text = uncomment(path.read_text())
        sources[relative] = text
        for value in re.findall(r'\\input\{([^}]+)\}', text):
            target = REPORTS / value
            if not target.suffix:
                target = target.with_suffix('.tex')
            inputs.append({'source': relative, 'file': value, 'exists': target.is_file()})
            if not target.is_file():
                errors.append('Missing input: ' + value)
            elif value != 'appendices/generated.tex':
                inspect(target)
        for value in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', text):
            target = REPORTS / value
            row = {'file': value, 'exists': target.is_file()}
            if target.is_file():
                row.update(sha256=digest(target), bytes=target.stat().st_size)
                row.update(raster_info(target))
                if row['format'] == 'unknown':
                    errors.append('Unknown image format: ' + value)
            else:
                errors.append('Missing graphic: ' + value)
            graphics.append(row)
        if re.search(r'/(?:home|tmp|file_storage01)/', text):
            errors.append('Server absolute path in ' + relative)
        if re.search(r'\\pending\b|TODO|待补充', text):
            errors.append('Unfilled body placeholder in ' + relative)
        depth = 0
        for token in re.findall(r'(?<!\\)[{}]', text):
            depth += 1 if token == '{' else -1
            if depth < 0:
                errors.append('Unexpected closing brace in ' + relative)
                break
        if depth:
            errors.append('Unbalanced braces in ' + relative)
        envs = []
        for command, env in re.findall(r'\\(begin|end)\{([^}]+)\}', text):
            if command == 'begin':
                envs.append(env)
            elif not envs or envs.pop() != env:
                errors.append('Environment mismatch in ' + relative + ': ' + env)
        if envs:
            errors.append('Unclosed environments in ' + relative + ': ' + str(envs))
        for env, body in re.findall(r'\\begin\{(figure|table)\}(.*?)\\end\{\1\}', text, re.S):
            if '\\caption{' not in body or '\\label{' not in body:
                errors.append('Missing float caption/label in ' + relative)

    inspect(REPORTS / 'main.tex')
    body = '\n'.join(sources.values())
    labels = re.findall(r'\\label\{([^}]+)\}', body)
    duplicates = [k for k, v in collections.Counter(labels).items() if v > 1]
    errors.extend('Duplicate label: ' + k for k in duplicates)
    refs = re.findall(r'\\(?:eqref|ref|pageref)\{([^}]+)\}', body)
    errors.extend('Missing label: ' + k for k in refs if k not in labels)
    # Float labels should be explained in the text, not merely attached to a caption.
    errors.extend('Unreferenced float: ' + k for k in labels if k.startswith(('fig:', 'tab:')) and k not in refs)
    bib = (REPORTS / 'references.bib').read_text()
    keys = re.findall(r'@\w+\s*\{\s*([^,]+),', bib)
    citations = []
    for group in re.findall(r'\\cite\w*\{([^}]+)\}', body):
        citations.extend(x.strip() for x in group.split(','))
    errors.extend('Missing BibTeX key: ' + k for k in citations if k not in keys)
    errors.extend('Uncited BibTeX key: ' + k for k in keys if k not in citations)
    if len(keys) != len(set(keys)):
        errors.append('Duplicate BibTeX keys')
    bibdepth = 0
    for token in re.findall(r'(?<!\\)[{}]', bib):
        bibdepth += 1 if token == '{' else -1
        if bibdepth < 0:
            errors.append('BibTeX unexpected closing brace')
            break
    if bibdepth:
        errors.append('Unbalanced BibTeX braces')
    for key in keys:
        entry = re.search(r'@\w+\s*\{\s*' + re.escape(key) + r',(.*?)(?=\n@|\Z)', bib, re.S).group(1)
        for field in ['author', 'title', 'year', 'url']:
            if not re.search(r'\b' + field + r'\s*=', entry):
                errors.append('Missing bibliography field ' + field + ': ' + key)
    if len(re.findall(r'\\section\{', body)) != 9:
        errors.append('Expected nine body sections')
    original_figure = ROOT / 'third_party/GraphMAE/imgs/fig.png'
    figure_copy = REPORTS / 'figures/graphmae_official.jpg'
    if not figure_copy.exists() or digest(figure_copy) != digest(original_figure):
        errors.append('Official figure copy must remain byte-identical')
    # Historical manifest contains some now-authorized report/management changes.
    # Protect experiment/data/figure/code/annex materials, not main.tex or decisions.
    protected = {}
    for stage in ['M1', 'M2']:
        entries = json.loads((ROOT / f'results/processed/{stage}/artifact_manifest.json').read_text())['artifacts']
        pairs = entries.items() if isinstance(entries, dict) else [(x['file'], x) for x in entries]
        for file, record in pairs:
            if file.startswith(('results/', 'figures/', 'scripts/', 'configs/', 'data/', 'third_party/', 'agent_logs/', 'docs/paper/', 'reports/appendices/')):
                protected[file] = record['sha256']
    changed = [p for p, h in protected.items() if not (ROOT / p).is_file() or digest(ROOT / p) != h]
    errors.extend('Protected evidence changed: ' + p for p in changed)
    repos = []
    for name in ['GraphMAE', 'GraphMAE-main-audit']:
        path = ROOT / 'third_party' / name
        head = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
        status = subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain'], text=True).strip()
        repos.append({'directory': 'third_party/' + name, 'commit': head, 'clean': not status})
        expected = '9127f25e01aa5ba1cca68b746284aeb7ab2b4850' if name == 'GraphMAE' else 'b14f080c919257b495e3cb6474286d5252d6a635'
        if status or head != expected:
            errors.append('Official tree/commit changed: ' + name)
    report = {
        'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status': 'passed_static_only' if not errors else 'failed',
        'latex_compilation_attempted': False, 'model_execution_attempted': False,
        'scope': 'Nine body sections and TikZ checked; appendix whole-file paths/completeness checked separately by package_delivery.py.',
        'limitations': 'Regex/stack checks do not validate TeX expansion, fonts, table widths, float pagination, PDF output, or bibliography rendering.',
        'body_sections': 9, 'figures': len(re.findall(r'\\begin\{figure\}', body)),
        'tables': len(re.findall(r'\\begin\{table\}', body)),
        'numbered_math_environments': len(re.findall(r'\\begin\{(?:equation|align)\}', body)),
        'labels': len(labels), 'references': len(refs), 'bibliography_keys': keys,
        'citation_order': list(dict.fromkeys(citations)), 'inputs': inputs, 'graphics': graphics,
        'han_characters_in_sources': len(re.findall(r'[\u4e00-\u9fff]', body)),
        'source_sha256': {**{p: digest(ROOT / p) for p in sources}, 'reports/references.bib': digest(REPORTS / 'references.bib')},
        'protected_artifacts_checked': len(protected), 'protected_mismatches': changed,
        'official_repositories': repos, 'errors': errors}
    (REPORTS / output_name).write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['status', 'body_sections', 'figures', 'tables', 'labels', 'han_characters_in_sources', 'protected_artifacts_checked', 'errors']}, ensure_ascii=False))
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
