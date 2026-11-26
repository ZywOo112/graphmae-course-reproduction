"""Snapshot evidence, check final appendices, and create one portable ZIP.

No model/TeX execution. Only project delivery files are written. Existing final
ZIPs are never overwritten. Run snapshot before making delivery edits.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / 'reports/M3_3_PROTECTED.json'
FONT_SUFFIXES = {'.otf', '.ttf', '.ttc', '.woff', '.woff2', '.pfb', '.pfa', '.eot', '.dfont', '.fon'}
SKIP_DIRS = {'.venv', '__pycache__', '.git', '.cache', 'deliverables'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def files():
    for base, dirs, names in os.walk(ROOT, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for name in sorted(names):
            p = Path(base) / name
            if name == '.git' or p.suffix == '.pyc':
                continue
            if p.is_symlink():
                raise ValueError('Symlink requires explicit review: ' + str(p))
            yield p


def snapshot():
    assert not SNAPSHOT.exists(), 'Existing protection snapshot must not be replaced'
    prefixes = ('results/', 'data/', 'figures/', 'third_party/', 'agent_logs/M0/')
    targets = {p.relative_to(ROOT).as_posix(): digest(p) for p in files()
               if p.relative_to(ROOT).as_posix().startswith(prefixes)}
    sections = {p.relative_to(ROOT).as_posix(): digest(p) for p in (ROOT / 'reports/sections').glob('*.tex')}
    write_json(SNAPSHOT, {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                          'sha256': targets, 'body_before': sections,
                          'policy': 'All historical evidence preserved; only chapter 7 table typography may change.'})
    print(json.dumps({'protected_files': len(targets)}, ensure_ascii=False))


def verify():
    from generate_appendices import discover
    from export_agent_logs import load_records, select_tasks
    errors = []
    before = json.loads(SNAPSHOT.read_text())
    changed = [f for f, h in before['sha256'].items() if not (ROOT / f).is_file() or digest(ROOT / f) != h]
    errors.extend('Historical evidence changed: ' + f for f in changed)
    body_changed = [f for f, h in before['body_before'].items() if digest(ROOT / f) != h]
    if set(body_changed) - {'reports/sections/07_results.tex'}:
        errors.append('Unexpected body changes: ' + str(body_changed))
    manifest = json.loads((ROOT / 'reports/appendices/manifest.json').read_text())
    expected = [p.relative_to(ROOT).as_posix() for p in discover()[0]]
    if manifest['sources'] != expected:
        errors.append('Self-written code/config discovery differs from appendix manifest')
    imports = re.findall(r'\\appendix(?:file|text)\{\.\./([^}]+)\}', (ROOT / 'reports/appendices/generated.tex').read_text())
    expected_imports = manifest['sources'] + manifest['records'] + manifest['workflows']
    if imports != expected_imports or len(imports) != len(set(imports)):
        errors.append('Appendix whole-file imports differ/duplicate')
    for f, h in manifest['sha256'].items():
        if not (ROOT / f).exists() or digest(ROOT / f) != h:
            errors.append('Appendix hash mismatch: ' + f)
    if any('commentary' in f for f in imports) or any(f.endswith('generated.tex') for f in imports):
        errors.append('Forbidden commentary or recursive import')
    for relative in imports:
        if not (ROOT / relative).is_file():
            errors.append('Missing verbatim input: ' + relative)
    records = load_records()
    session = Path(records['session_file'])
    tasks = select_tasks(session)
    for stage, record in records['stages'].items():
        actual = tasks[stage]
        for name, value in [('prompt.md', actual['prompt']), ('response.md', actual['response'])]:
            target = ROOT / 'agent_logs' / stage / name
            if value is None or target.read_bytes() != value.encode('utf-8'):
                errors.append('Not identical to real session: ' + stage + '/' + name)
    main = (ROOT / 'reports/main.tex').read_text()
    if '\\includeFullAppendixtrue' not in main:
        errors.append('Full appendices are not enabled')
    generated = (ROOT / 'reports/appendices/generated.tex').read_text()
    if len(re.findall(r'\\section\{', generated)) != 4:
        errors.append('Expected appendices A-D')
    for option in ['firstline=', 'lastline=', '\\newpage', '\\clearpage', '\\begin{minipage}']:
        if option in generated:
            errors.append('Truncation/forced file pagination in appendix: ' + option)
    report = {'status': 'passed_static_only' if not errors else 'failed',
              'latex_compiled': False, 'models_rerun': False,
              'historical_files_checked': len(before['sha256']), 'historical_mismatches': changed,
              'body_style_changes': body_changed, 'printed_code_config_files': len(expected),
              'electronic_report_sources': len(manifest['electronic_report_sources']),
              'whole_file_verbatim_imports': len(imports), 'real_stages': list(records['stages']),
              'records_verified_against_original_session': True, 'errors': errors}
    write_json(ROOT / 'reports/M3_3_STATIC_CHECK.json', report)
    assert not errors, errors
    print(json.dumps(report, ensure_ascii=False))
    return report


def payload():
    result = []
    for p in files():
        relative = p.relative_to(ROOT).as_posix()
        if p.suffix.lower() in FONT_SUFFIXES:
            continue
        if relative == 'reports/M3_3_DELIVERY.json':
            continue  # external receipt covers the whole ZIP; avoid hash recursion
        if relative.startswith('agent_logs/') and p.name in {'commentary.md', 'commentary.json'}:
            continue  # keep original M0 files on server, but submit main tasks only
        if relative.startswith('reports/') and p.suffix in {'.aux', '.bbl', '.blg', '.fls', '.fdb_latexmk', '.synctex'}:
            continue
        result.append(p)
    return sorted(result)


def package():
    verification = verify()
    out = ROOT / 'deliverables'
    out.mkdir(exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    archive = out / ('GraphMAE_M3_3_' + stamp + '.zip')
    temporary = archive.with_suffix('.partial')
    assert not archive.exists() and not temporary.exists()
    acceptance = ROOT / 'reports/M3_3_ACCEPTANCE.md'
    acceptance.write_text(re.sub(r'交付ZIP：.*?。', '交付ZIP：`' + archive.relative_to(ROOT).as_posix() + '`。', acceptance.read_text()), encoding='utf-8')
    # Only the rounded size is printed inside the ZIP; exact bytes/hash live in
    # the external receipt, avoiding an impossible self-hash. Stabilize once.
    for attempt in range(4):
        candidates = payload()
        records = {p.relative_to(ROOT).as_posix(): {'bytes': p.stat().st_size, 'sha256': digest(p)} for p in candidates}
        index = {'root': 'GraphMAE/', 'files': records,
                 'exclusions': ['.venv', '__pycache__', '.git pointers/objects', 'font binaries', 'commentary exports', 'delivery ZIP/receipt'],
                 'integrity': 'FILE_MANIFEST lists all payload files. SHA256SUMS additionally covers FILE_MANIFEST. External ZIP SHA256 covers both index files.'}
        manifest = (json.dumps(index, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
        sums = ''.join(v['sha256'] + '  ' + k + '\n' for k, v in records.items())
        sums += hashlib.sha256(manifest).hexdigest() + '  FILE_MANIFEST.json\n'
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for p in candidates:
                z.write(p, 'GraphMAE/' + p.relative_to(ROOT).as_posix())
            z.writestr('GraphMAE/FILE_MANIFEST.json', manifest)
            z.writestr('GraphMAE/SHA256SUMS.txt', sums.encode('utf-8'))
        size_label = f'{temporary.stat().st_size / 1024**2:.1f} MiB'
        text = acceptance.read_text()
        revised = re.sub(r'ZIP压缩大小：.*?。', 'ZIP压缩大小：约' + size_label + '。', text)
        if revised == text:
            break
        acceptance.write_text(revised, encoding='utf-8')
    else:
        raise RuntimeError('Rounded archive size failed to stabilize')
    # Verify every byte in the archive without extracting a duplicate project.
    with zipfile.ZipFile(temporary) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == len(records) + 2
        for file, record in records.items():
            data = z.read('GraphMAE/' + file)
            assert len(data) == record['bytes'] and hashlib.sha256(data).hexdigest() == record['sha256'], file
            assert digest(ROOT / file) == record['sha256'], 'Project changed while packaging: ' + file
        assert all(Path(n).suffix.lower() not in FONT_SUFFIXES for n in z.namelist())
        assert not any(any(part in SKIP_DIRS for part in Path(n).parts) for n in z.namelist())
        # Validate all types of report input using ZIP entries, resolving from reports/.
        for name in z.namelist():
            if name.startswith('GraphMAE/reports/') and name.endswith('.tex'):
                content = z.read(name).decode('utf-8')
                for value in re.findall(r'\\(?:input|appendixfile|appendixtext)\{([^}]+)\}|\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', content):
                    value = next(x for x in value if x)
                    if value.startswith('#'):
                        continue  # macro argument, not a literal filename
                    target = os.path.normpath('GraphMAE/reports/' + value).replace('\\', '/')
                    if not Path(target).suffix:
                        target += '.tex'
                    assert target in z.namelist(), 'Missing ZIP TeX dependency: ' + target
        for repo in ['GraphMAE', 'GraphMAE-main-audit']:
            import subprocess
            tracked = subprocess.check_output(['git', '-C', str(ROOT / 'third_party' / repo), 'ls-files'], text=True).splitlines()
            assert all('GraphMAE/third_party/' + repo + '/' + f in z.namelist() for f in tracked)
    temporary.rename(archive)
    zip_hash = digest(archive)
    archive.with_suffix('.zip.sha256').write_text(zip_hash + '  ' + archive.name + '\n', encoding='ascii')
    receipt = {'status': 'passed', 'archive': archive.relative_to(ROOT).as_posix(),
               'archive_bytes': archive.stat().st_size, 'archive_mib': archive.stat().st_size / 1024**2,
               'sha256': zip_hash, 'payload_files': len(records), 'zip_entries': len(records) + 2,
               'uncompressed_payload_bytes': sum(r['bytes'] for r in records.values()),
               'all_zip_hashes_and_relative_inputs_verified': True,
               'all_upstream_tracked_files_included': True, 'fonts_and_caches_excluded': True,
               'latex_compiled': False, 'printed_code_config_files': verification['printed_code_config_files']}
    write_json(ROOT / 'reports/M3_3_DELIVERY.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['snapshot', 'verify', 'package'])
    action = parser.parse_args().action
    {'snapshot': snapshot, 'verify': verify, 'package': package}[action]()
