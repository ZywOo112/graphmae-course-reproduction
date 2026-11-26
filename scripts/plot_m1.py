"""Plot only real completed M1 data; use existing Matplotlib and a local CJK font."""
import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ft2font import FT2Font
    import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', type=Path, default=ROOT / 'results/processed/M1/summary.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'figures/M1')
    parser.add_argument('--archive-existing', action='store_true', help='Keep previous derived plots in a timestamped archive before rendering')
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    assert summary['status'] == 'passed' and summary['seeds'] == list(range(5))
    destination = args.output
    destination.mkdir(parents=True, exist_ok=True)
    filenames = ['loss_curves', 'loss_mean_std', 'accuracy_comparison', 'validation_results', 'runtime_comparison']
    if any((destination / (name + '.png')).exists() for name in filenames):
        if not args.archive_existing:
            raise RuntimeError('Existing figures; choose a fresh --output or explicitly archive previous derived plots')
        archive = destination / ('previous_render_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
        archive.mkdir()
        for name in [value + '.png' for value in filenames] + ['plot_metadata.json']:
            if (destination / name).exists():
                (destination / name).rename(archive / name)
    required_characters = set('五个随机种子的预训练损失曲线均值样本标准差测试准确率比较最佳验证与选中轮次运行时间秒其他开销周期最终自监督完整')
    font_path = None
    for font in font_manager.fontManager.ttflist:
        try:
            cmap = FT2Font(font.fname).get_charmap()
            if all(ord(character) in cmap for character in required_characters):
                font_path = Path(font.fname)
                break
        except RuntimeError:
            continue
    font_provenance = {'system_font': str(font_path) if font_path else None}
    if font_path is None:
        assets = destination / 'assets'
        assets.mkdir(exist_ok=True)
        font_path = assets / 'NotoSansCJKsc-Regular.otf'
        urls = {'NotoSansCJKsc-Regular.otf': 'https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf', 'LICENSE': 'https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/LICENSE'}
        for name, url in urls.items():
            path = assets / name
            if not path.exists():
                with urllib.request.urlopen(url, timeout=90) as response:
                    data = response.read()
                with path.open('xb') as handle:
                    handle.write(data)
        font_provenance = {'source_urls': urls, 'sha256': {name: hashlib.sha256((assets / name).read_bytes()).hexdigest() for name in urls}, 'font_file_mtime_utc': datetime.datetime.fromtimestamp(font_path.stat().st_mtime, datetime.timezone.utc).isoformat(), 'installation': 'project-local asset only; no system or Python environment changed'}
    assert all(ord(character) in FT2Font(str(font_path)).get_charmap() for character in required_characters)
    font_manager.fontManager.addfont(str(font_path))
    family = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update({'font.family': family, 'axes.unicode_minus': False, 'font.size': 11, 'axes.titlesize': 14, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 300})
    records = summary['records']
    epochs = np.arange(1, 1501)
    curves = []
    for record in records:
        with (ROOT / record['result_directory'] / 'training.csv').open() as handle:
            curve = [float(row['loss']) for row in csv.DictReader(handle)]
        assert len(curve) == 1500 and np.isfinite(curve).all()
        curves.append(curve)
    curves = np.array(curves)
    saved = []
    def finish(figure, name):
        figure.tight_layout()
        path = destination / (name + '.png')
        figure.savefig(path, dpi=300, facecolor='white', bbox_inches='tight')
        plt.close(figure)
        saved.append({'file': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    figure, axis = plt.subplots(figsize=(10, 5.5))
    for seed, values in enumerate(curves):
        axis.plot(epochs, values, label=f'种子 {seed}', alpha=0.7, linewidth=0.75)
    axis.set(title='五个随机种子的 GraphMAE 预训练损失', xlabel='预训练轮次', ylabel='SCE 损失')
    axis.legend(ncol=5)
    axis.grid(alpha=0.2)
    finish(figure, 'loss_curves')
    figure, axis = plt.subplots(figsize=(10, 5.5))
    mean, sd = curves.mean(0), curves.std(0, ddof=1)
    axis.plot(epochs, mean, color='#206496', label='五种子均值', linewidth=1)
    axis.fill_between(epochs, mean - sd, mean + sd, color='#57a7d3', alpha=0.35, label='均值 ± 样本标准差（ddof=1）')
    axis.set(title='预训练损失的均值与种子间波动', xlabel='预训练轮次', ylabel='SCE 损失')
    axis.legend()
    axis.grid(alpha=0.2)
    finish(figure, 'loss_mean_std')
    seeds = np.arange(5)
    accuracy = np.array([record['test_accuracy'] * 100 for record in records])
    figure, axis = plt.subplots(figsize=(8, 5.5))
    bars = axis.bar(seeds, accuracy, color='#347eae', width=0.6, label='本项目：最终测试准确率')
    axis.bar_label(bars, labels=[f'{value:.1f}%' for value in accuracy], padding=3)
    axis.axhline(84.2, color='#c56b34', linestyle='--', label='论文均值 84.2%（20次初始化）')
    axis.set(title='Cora 五个种子的最终测试准确率', xlabel='随机种子', ylabel='准确率（%）', ylim=(0, 100), xticks=seeds)
    axis.legend(loc='lower right')
    finish(figure, 'accuracy_comparison')
    figure, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    bars = axes[0].bar(seeds, [r['best_val_accuracy'] * 100 for r in records], color='#3d9b80')
    axes[0].bar_label(bars, fmt='%.1f%%', padding=3)
    axes[0].set(title='最佳验证准确率', xlabel='随机种子', ylabel='准确率（%）', ylim=(0, 100), xticks=seeds)
    selected = [r['selected_epoch'] for r in records]
    bars = axes[1].bar(seeds, selected, color='#8972b8')
    axes[1].bar_label(bars, labels=[str(value) for value in selected], padding=3)
    axes[1].set(title='验证集选中的分类器轮次', xlabel='随机种子', ylabel='线性评估轮次（从1计数）', ylim=(0, 330), xticks=seeds)
    figure.suptitle('最终线性评估：验证结果与选中轮次', y=0.98)
    finish(figure, 'validation_results')
    figure, axis = plt.subplots(figsize=(9, 5.5))
    components = [np.array([r[key] for r in records]) for key in ['pretrain_seconds', 'periodic_probe_seconds', 'final_probe_seconds']]
    totals = np.array([r['runtime_seconds'] for r in records])
    components.append(np.maximum(totals - sum(components), 0))
    bottom = np.zeros(5)
    for values, label, color in zip(components, ['自监督预训练', '周期验证 Probe', '最终线性评估', '加载、保存及其他开销'], ['#347eae', '#6eb79f', '#9b82be', '#b8bcc4']):
        axis.bar(seeds, values, bottom=bottom, label=label, color=color, width=0.6)
        bottom += values
    for seed, total in enumerate(totals):
        axis.text(seed, total + 0.3, f'{total:.2f}', ha='center')
    axis.set(title='五个种子的运行时间', xlabel='随机种子', ylabel='时间（秒）', xticks=seeds, ylim=(0, max(totals) * 1.3))
    axis.legend(loc='upper left', ncol=2)
    finish(figure, 'runtime_comparison')
    metadata = {'summary': str(args.summary), 'batch': summary['batch'], 'dpi': 300, 'font': family, 'font_provenance': font_provenance, 'matplotlib': matplotlib.__version__, 'numpy': np.__version__, 'loss_band': 'pointwise sample SD across five seeds, ddof=1, not confidence interval', 'figures': saved}
    with (destination / 'plot_metadata.json').open('x') as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps(metadata, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
