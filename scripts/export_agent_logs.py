"""One-shot real primary task/final export; no watcher or historical writes."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADINGS = {
    'M0': '# GraphMAE 课程大作业：M0 项目初始化与可行性审计',
    'M1': '# GraphMAE 机器学习课程大作业：M1 原始算法正式复现',
    'M2': '# GraphMAE 机器学习课程大作业：M2 算法改进与自主实验',
    'M3_1': '# GraphMAE 机器学习课程大作业：M3 第一阶段——材料盘点与报告结构设计',
    'M3_2': '# GraphMAE 机器学习课程大作业：M3.2 正式论文写作与图表制作',
}

def text_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def select_tasks(session):
    tasks = {}
    active = None
    for number, line in enumerate(session.open(encoding='utf-8'), 1):
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        p = item.get('payload', {})
        if item.get('type') != 'response_item' or p.get('type') != 'message':
            continue
        body = ''.join(c.get('text', '') for c in p.get('content', [])
                       if c.get('type') in {'input_text', 'output_text'})
        if p.get('role') == 'user':
            if body.startswith(('<environment_context>', '<codex_internal_context')):
                continue
            active = None
            for stage, heading in HEADINGS.items():
                if heading in body[:700].splitlines():
                    if stage in tasks:
                        raise ValueError('Duplicate task: ' + stage)
                    tasks[stage] = {'prompt': body, 'response': None,
                                    'prompt_line': number, 'prompt_utc': item.get('timestamp')}
                    active = stage
                    break
        elif active and p.get('role') == 'assistant' and p.get('phase') in {'final', 'final_answer'}:
            task = tasks[active]
            if task['response'] is not None:
                if task['response'] == body:
                    continue
                raise ValueError('Multiple finals: ' + active)
            task.update(response=body, response_line=number, response_utc=item.get('timestamp'))
    if set(HEADINGS) - set(tasks):
        raise ValueError('Missing major task: ' + str(set(HEADINGS) - set(tasks)))
    return tasks

def immutable_write(path, body):
    raw = body.encode('utf-8')
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError('Existing real record differs: ' + str(path))
    else:
        path.write_bytes(raw)

def load_records():
    return json.loads((ROOT / 'agent_logs/MAIN_RECORDS.json').read_text(encoding='utf-8'))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    args = parser.parse_args()
    tasks = select_tasks(args.session)
    records = {'session_file': str(args.session.resolve()),
               'policy': 'Exact primary user tasks and assistant finals only. No M3.3 before its final, no commentary or unrelated projects.',
               'stages': {}}
    for stage in HEADINGS:
        task = tasks[stage]
        directory = ROOT / 'agent_logs' / stage
        directory.mkdir(exist_ok=True)
        immutable_write(directory / 'prompt.md', task['prompt'])
        if task['response'] is not None:
            immutable_write(directory / 'response.md', task['response'])
        records['stages'][stage] = {
            'status': 'complete' if task['response'] is not None else 'missing_real_final',
            'prompt': f'agent_logs/{stage}/prompt.md',
            'response': f'agent_logs/{stage}/response.md' if task['response'] is not None else None,
            'prompt_sha256': text_hash(task['prompt']),
            'response_sha256': text_hash(task['response']) if task['response'] is not None else None,
            **{k: v for k, v in task.items() if k not in {'prompt', 'response'}}}
    (ROOT / 'agent_logs/MAIN_RECORDS.json').write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: {'status': v['status'], 'prompt_sha256': v['prompt_sha256'], 'response_sha256': v['response_sha256']} for k, v in records['stages'].items()}, ensure_ascii=False))

if __name__ == '__main__':
    main()
