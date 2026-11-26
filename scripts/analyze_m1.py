"""Validate all five formal runs and aggregate them with sample SD (ddof=1)."""
import argparse
import csv
import datetime
import json
from pathlib import Path
import sys

from run_m1 import ROOT, REPO, COMMIT, validate_result, save_json, status_file
from preflight_m1 import digest, execute

def main():
    import numpy as np
    import torch
    from sklearn.metrics import accuracy_score, f1_score
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/processed/M1')
    args = parser.parse_args()
    batch = args.batch.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ['summary.csv', 'summary.json', 'comparison.md']):
        raise RuntimeError('Existing analysis output; choose a fresh --output directory')
    preflight = json.loads((batch / 'preflight.json').read_text())
    assert execute(['git', '-C', REPO, 'rev-parse', 'HEAD']).strip() == COMMIT
    assert not execute(['git', '-C', REPO, 'status', '--porcelain']).strip()
    for name, value in preflight['data']['sha256'].items():
        assert digest(ROOT / 'data' / name) == value
    raw_data, _ = torch.load(ROOT / 'data/cora/processed/data.pt', weights_only=False)
    records = []
    reference_config = None
    loss_curves = []
    all_checks = []
    for seed in range(5):
        directory = batch / f'seed_{seed}'
        result = validate_result(directory)
        config = json.loads((directory / 'effective_config.json').read_text())
        assert config['seed'] == seed and config['official_commit'] == COMMIT
        for name, value in config['runner_sources_sha256'].items():
            assert digest(ROOT / 'scripts' / name) == value
        comparable = {name: value for name, value in config['official_args'].items() if name != 'seeds'}
        if reference_config is None:
            reference_config = comparable
        assert comparable == reference_config
        assert config['data']['sha256'] == preflight['data']['sha256']
        assert config['rng']['cudnn_deterministic'] and not config['rng']['cudnn_benchmark']
        assert not config['rng']['deterministic_algorithms']
        assert config['gpu_mapping']['physical_uuid'] == preflight['selected_gpu']['uuid']
        prediction = np.load(directory / 'test_predictions.npz')
        assert np.array_equal(prediction['node_id'], np.flatnonzero(raw_data.test_mask.numpy()))
        assert np.array_equal(prediction['target'], raw_data.y.numpy()[prediction['node_id']])
        recomputed_accuracy = float(accuracy_score(prediction['target'], prediction['prediction']))
        recomputed_f1 = float(f1_score(prediction['target'], prediction['prediction'], average='micro'))
        assert recomputed_accuracy == result['test_accuracy'] and recomputed_f1 == result['test_micro_f1']
        with (directory / 'training.csv').open() as handle:
            losses = list(csv.DictReader(handle))
        loss_curves.append([float(row['loss']) for row in losses])
        periodic = json.loads((directory / 'periodic_probes.json').read_text())
        assert [p['pretrain_epoch'] for p in periodic] == list(range(200, 1500, 200))
        for probe in periodic:
            assert not probe['test_evaluated'] and probe['encoder_unchanged']
            with (directory / 'probes' / f"epoch_{probe['pretrain_epoch']:04d}_history.csv").open() as handle:
                history = list(csv.DictReader(handle))
            assert len(history) == 300 and max(float(row['val_accuracy']) for row in history) == probe['best_val_accuracy']
        pretrained = torch.load(directory / 'pretrained_final.pt', map_location='cpu', weights_only=False)
        classifier = torch.load(directory / 'classifier_best_val.pt', map_location='cpu', weights_only=False)
        assert pretrained['epoch'] == 1500 and pretrained['seed'] == seed
        assert tuple(classifier['state_dict']['linear.weight'].shape) == (7, 512)
        assert all(torch.isfinite(value).all() for value in pretrained['state_dict'].values())
        record = {'seed': seed, 'best_val_accuracy': result['best_val_accuracy'],
                  'selected_epoch': result['selected_epoch'],
                  'selected_epoch_zero_based': result['selected_epoch_zero_based'],
                  'test_accuracy': recomputed_accuracy, 'micro_f1': recomputed_f1,
                  'runtime_seconds': result['total_seconds'],
                  'pretrain_seconds': result['pretrain_seconds_excluding_periodic_probes'],
                  'periodic_probe_seconds': result['periodic_probe_seconds'],
                  'final_probe_seconds': result['final_probe_seconds'],
                  'peak_allocated_mib': result['peak_allocated_mib'],
                  'peak_reserved_mib': result['peak_reserved_mib'],
                  'result_directory': str(directory.relative_to(ROOT))}
        records.append(record)
        all_checks.append({'seed': seed, 'complete_1500_and_8x300': True, 'prediction_metrics_independently_recomputed': True, 'labels_and_split_match_cached_Cora': True, 'checkpoint_integrity_and_shapes': True, 'no_test_in_periodic_or_final_probe_selection': True, 'parameters_config_and_sources': True})
    loss_array = np.array(loss_curves)
    assert loss_array.shape == (5, 1500) and np.isfinite(loss_array).all()
    accuracy_values = np.array([r['test_accuracy'] for r in records])
    f1_values = np.array([r['micro_f1'] for r in records])
    summary = {'stage': 'M1', 'status': 'passed', 'batch': str(batch.relative_to(ROOT)),
               'official_commit': COMMIT, 'seeds': list(range(5)), 'n_seeds': 5,
               'accuracy_mean': float(accuracy_values.mean()), 'accuracy_sample_std': float(accuracy_values.std(ddof=1)),
               'micro_f1_mean': float(f1_values.mean()), 'micro_f1_sample_std': float(f1_values.std(ddof=1)),
               'std_ddof': 1, 'average_runtime_seconds': float(np.mean([r['runtime_seconds'] for r in records])),
               'total_seed_runtime_seconds': float(sum(r['runtime_seconds'] for r in records)),
               'paper_reference': {'accuracy_mean': 0.842, 'reported_std': 0.004, 'n_initializations': 20, 'source': 'GraphMAE arXiv v3 Table 1; reported reference, not a local run'},
               'difference_from_paper_percentage_points': float((accuracy_values.mean() - 0.842) * 100),
               'selected_epoch_numbering': 'one-based; zero-based field separately provided',
               'accuracy_micro_f1_identity': 'same single-label multiclass predictions, not independent evidence',
               'protocol': {'pretrain_epochs_per_seed': 1500, 'periodic_probe_epochs': list(range(200, 1500, 200)), 'classifier_epochs_per_probe': 300, 'final_encoder_selection': 'fixed final epoch; periodic probes never choose encoder', 'classifier_selection': 'final probe validation accuracy, later tie', 'test_evaluations_per_seed': 1},
               'records': records}
    with (output / 'summary.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    save_json(output / 'summary.json', summary)
    with (output / 'loss_statistics.csv').open('x', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['epoch', 'loss_mean', 'loss_sample_std'])
        for index, (mean, sd) in enumerate(zip(loss_array.mean(0), loss_array.std(0, ddof=1)), start=1):
            writer.writerow([index, mean, sd])
    save_json(output / 'quality_checks.json', {'status': 'passed', 'checks': all_checks, 'M0_data_sha256_unchanged': True, 'official_git_worktree_clean': True, 'no_hyperparameter_search_or_seed_selection': True})
    rows = ['# M1 Cora 原始算法正式复现结果', '', f"原始批次：`{summary['batch']}`。五个指定种子均完整完成，不含M0短测。", '', '| Seed | Best Val Accuracy | Selected Epoch（从1计数） | Test Accuracy | Micro-F1 | Runtime（秒） |', '|---|---|---|---|---|---|']
    rows += [f"| {r['seed']} | {r['best_val_accuracy']*100:.2f}% | {r['selected_epoch']} | {r['test_accuracy']*100:.2f}% | {r['micro_f1']*100:.2f}% | {r['runtime_seconds']:.2f} |" for r in records]
    rows += ['', f"Test Accuracy：**{summary['accuracy_mean']*100:.2f} ± {summary['accuracy_sample_std']*100:.2f}%**（5种子，样本标准差ddof=1）。", f"Micro-F1：{summary['micro_f1_mean']*100:.2f} ± {summary['micro_f1_sample_std']*100:.2f}%；与Accuracy来自相同预测，不构成两项独立提升证据。", f"平均每种子运行时间：{summary['average_runtime_seconds']:.2f}秒，五次合计{summary['total_seed_runtime_seconds']:.2f}秒。时间包含worker导入、数据加载、模型初始化、预训练、8次Probe及结果保存至指标记录；不含共享preflight和父进程质量复核。", '', '## 与论文的参考性比较', '', '论文表1报告Cora Accuracy为84.2±0.4%，使用20次初始化；本项目仅5个指定种子，且样本标准差口径明确为ddof=1。±不是置信区间，不能据此声称统计等价或显著提升。', f"本项目均值相对论文差{summary['difference_from_paper_percentage_points']:+.2f}个百分点；固定参数与全部种子均保留，未为接近论文结果调参或挑种子。", '', '| 差异 | 论文 | 本项目 |', '|---|---|---|', '| 重复次数 | 20次初始化 | seeds 0–4，共5次 |', '| 软件 | PyTorch1.9、DGL0.7.2、Python3.7、CUDA10.2 | 官方PyG分支；PyTorch2.4.1+cu121、PyG2.3.1、Python3.11.7 |', '| 硬件 | 论文附录一般实验环境NVIDIA2080Ti | 本次物理GPU2 NVIDIA A40；进程内cuda:0 |', '| 评估 | 原代码每Probe每轮计算test；只按val选择 | 周期Probe及最终Probe仅按train/val；冻结最终选择后仅一次测试 |', '| 随机性 | 原源码cuDNN属性拼写错误 | 包装器规范设种子、deterministic=True、benchmark=False；不强制确定性scatter |', '', '每200轮调用300轮周期Probe，保留CPU分类器初始化等RNG消耗，不隔离RNG。去掉test指标计算不影响优化更新或RNG；玩具用例验证通过，但GPU scatter仍可能非完全确定。所有预训练模型固定选择第1500轮，不根据周期验证结果选encoder。', '', '数据预处理、架构、官方Cora参数、完整训练轮数及验证集线性选模均已核对；无证据表明需要修改超参数。后续可进入另行授权的M2，当前不运行改进或Citeseer。', '', '图表由plot_m1.py直接读取该批次的全部五种子记录；loss波动带是跨种子逐轮样本标准差，不是置信区间。']
    with (output / 'comparison.md').open('x') as handle:
        handle.write('\n'.join(rows) + '\n')
    status_file(batch / 'batch_status.json', {'status': 'five_seeds_passed', 'seeds_completed': list(range(5)), 'quality_check': 'passed', 'analysis_directory': str(output), 'training_started': True})
    print(json.dumps({key: summary[key] for key in ['accuracy_mean', 'accuracy_sample_std', 'micro_f1_mean', 'micro_f1_sample_std', 'average_runtime_seconds', 'difference_from_paper_percentage_points']}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
