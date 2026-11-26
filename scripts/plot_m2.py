"""Render five Chinese 300dpi plots directly from completed M2 records."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', type=Path, default=ROOT / 'results/processed/M2/summary.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'figures/M2')
    cli = parser.parse_args()
    summary = json.loads(cli.summary.read_text())
    assert summary['status'] == 'passed' and summary['n_formal_runs'] == 20
    destination = cli.output
    destination.mkdir(parents=True, exist_ok=True)
    filenames = ['validation_comparison', 'paired_test_accuracy', 'degree_mask_rates', 'degree_classification', 'loss_curves']
    assert not any((destination / (name + '.png')).exists() for name in filenames), 'Use a fresh output directory instead of overwriting figures'
    font_path = ROOT / 'figures/M1/assets/NotoSansCJKsc-Regular.otf'
    font_manager.fontManager.addfont(str(font_path))
    family = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update({'font.family': family, 'font.size': 11, 'axes.titlesize': 14, 'axes.unicode_minus': False,
                         'axes.spines.top': False, 'axes.spines.right': False})
    selected = summary['selected_degree_alpha']
    records = []
    def finish(figure, name):
        figure.tight_layout()
        path = destination / (name + '.png')
        figure.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(figure)
        records.append({'file': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    alphas = [row['degree_alpha'] for row in summary['validation']]
    labels = ['均匀 Baseline' if alpha == 0 else f'度数系数 {alpha:g}' for alpha in alphas]
    means = [row['validation_accuracy_mean'] * 100 for row in summary['validation']]
    sd = [row['validation_accuracy_sample_std'] * 100 for row in summary['validation']]
    colors = ['#347eae', '#e8a257', '#7baf91', '#9a86ba']
    figure, axis = plt.subplots(figsize=(10, 5.5))
    bars = axis.bar(range(4), means, yerr=sd, capsize=5, color=colors)
    for index, (bar, mean) in enumerate(zip(bars, means)):
        axis.text(bar.get_x() + bar.get_width()/2, mean + sd[index] + 1, f'{mean:.2f}%', ha='center')
    axis.set(xticks=range(4), xticklabels=labels, ylim=(0, 100), xlabel='节点掩码采样策略', ylabel='验证准确率（%）',
             title=f'四组五种子验证结果（误差条为样本标准差）\n仅按验证集选定正向度数系数：{selected:g}')
    finish(figure, 'validation_comparison')
    figure, axis = plt.subplots(figsize=(8, 6))
    paired = summary['paired_records']
    values = []
    for row in paired:
        scores = [row['control_accuracy'] * 100, row['degree_accuracy'] * 100]
        values += scores
        axis.plot([0, 1], scores, marker='o', label=f"种子 {row['seed']}（差值{row['difference_percentage_points']:+.1f}个百分点）", linewidth=1.8)
    axis.set(xticks=[0, 1], xticklabels=['均匀 Baseline', f'选定度数系数 {selected:g}'], xlim=(-0.2, 1.2),
             ylim=(min(values) - 2, max(values) + 2), xlabel='掩码策略', ylabel='最终测试准确率（%）', title='五个种子的配对最终测试结果')
    axis.grid(axis='y', alpha=0.2)
    axis.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1.02, 1))
    finish(figure, 'paired_test_accuracy')
    x = np.arange(3)
    group_labels = ['低度数（1–2）', '中度数（3–4）', '高度数（≥5）']
    figure, axis = plt.subplots(figsize=(11, 6))
    width = 0.19
    for index, alpha in enumerate(alphas):
        rows = [row for row in summary['mask_groups'] if row['degree_alpha'] == alpha]
        positions = x + (index - 1.5) * width
        bars = axis.bar(positions, [row['mask_rate_mean'] * 100 for row in rows], width,
                        yerr=[row['mask_rate_sample_std'] * 100 for row in rows], capsize=2, color=colors[index], label=labels[index])
        axis.bar_label(bars, fmt='%.1f', padding=3, fontsize=8)
    axis.set(xticks=x, xticklabels=group_labels, ylim=(0, 100), xlabel='按唯一邻居数量确定的度数组',
             ylabel='组内节点实际掩码率（%）', title='不同度数组的真实掩码比例（五种子均值）')
    axis.legend(ncol=2, loc='upper left')
    finish(figure, 'degree_mask_rates')
    figure, axis = plt.subplots(figsize=(10, 6))
    for index, alpha in enumerate([0.0, selected]):
        rows = [row for row in summary['classification_groups'] if row['degree_alpha'] == alpha]
        bars = axis.bar(x + (index - 0.5) * 0.34, [row['accuracy_mean'] * 100 for row in rows], 0.34,
                        yerr=[row['accuracy_sample_std'] * 100 for row in rows], capsize=4,
                        label='均匀 Baseline' if alpha == 0 else f'选定度数系数 {selected:g}', color=colors[index])
        axis.bar_label(bars, fmt='%.2f%%', padding=4, fontsize=9)
    sizes = [row['test_nodes_per_seed'] for row in summary['classification_groups'] if row['degree_alpha'] == 0]
    axis.set(xticks=x, xticklabels=[f'{label}\n每种子测试节点数={count}' for label, count in zip(group_labels, sizes)], ylim=(0, 105),
             xlabel='预先固定的度数分组', ylabel='组内测试准确率（%）', title='不同度数节点的最终分类表现（误差条为样本标准差）')
    axis.legend(loc='lower right')
    finish(figure, 'degree_classification')
    figure, axis = plt.subplots(figsize=(11, 6))
    epochs = np.arange(1, 1501)
    batch = ROOT / summary['batch']
    for alpha, color in zip([0.0, selected], colors):
        name = 'control' if alpha == 0 else f'degree_alpha_{alpha:g}'
        curves = []
        for seed in range(5):
            with (batch / name / f'seed_{seed}/training.csv').open() as handle:
                curves.append([float(row['loss']) for row in csv.DictReader(handle)])
        curves = np.array(curves)
        assert curves.shape == (5, 1500) and np.isfinite(curves).all()
        mean, sd = curves.mean(0), curves.std(0, ddof=1)
        label = '均匀 Baseline' if alpha == 0 else f'选定度数系数 {selected:g}'
        axis.plot(epochs, mean, color=color, linewidth=1.0, label=label)
        axis.fill_between(epochs, mean - sd, mean + sd, color=color, alpha=0.2)
    axis.set(xlabel='预训练轮次', ylabel='SCE 重建损失', title='Baseline与选定方案的损失均值 ± 样本标准差\n掩码目标分布不同，Loss不等价于分类效果')
    axis.grid(alpha=0.2)
    axis.legend()
    finish(figure, 'loss_curves')
    with (destination / 'plot_metadata.json').open('x') as handle:
        json.dump({'summary': str(cli.summary), 'summary_sha256': hashlib.sha256(cli.summary.read_bytes()).hexdigest(),
                   'batch': summary['batch'], 'dpi': 300, 'font': family, 'font_file': str(font_path.relative_to(ROOT)),
                   'font_sha256': hashlib.sha256(font_path.read_bytes()).hexdigest(), 'font_license': 'figures/M1/assets/LICENSE',
                   'matplotlib': matplotlib.__version__, 'numpy': np.__version__, 'error_bars': 'sample SD across five seeds, ddof=1, not confidence interval',
                   'figures': records}, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print('Saved five M2 figures:', json.dumps(records, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
