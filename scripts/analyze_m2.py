"""Analyze sealed M2 experiments and saved predictions, without model inference."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

from run_m2 import ROOT, REPO, COMMIT, ALPHAS, arm, assert_inputs, check_paired_rng, validate_training, save_json
from preflight_m1 import digest, execute

def write_csv(path, records):
    with path.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import numpy as np
    import torch
    from sklearn.metrics import accuracy_score, f1_score
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/processed/M2')
    cli = parser.parse_args()
    batch = cli.batch.resolve()
    output = cli.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / 'summary.json').exists(), 'Existing analysis: use a fresh output directory'
    preflight = assert_inputs(batch)
    paired_rng = check_paired_rng(batch, False)
    selection_path = batch / 'alpha_selection.json'
    selection = json.loads(selection_path.read_text())
    selection_sha = digest(selection_path)
    selected = selection['selected_degree_alpha']
    degrees = np.load(batch / 'degree/degrees.npy')
    groups = np.load(batch / 'degree/groups.npy')
    metadata = json.loads((batch / 'degree/degree_metadata.json').read_text())
    raw_graph, _ = torch.load(ROOT / 'data/cora/processed/data.pt', weights_only=False)
    expected_test_ids = np.flatnonzero(raw_graph.test_mask.numpy())
    validation = []
    validation_grouped = []
    mask_rows = []
    all_reports = {}
    canonical_config = None
    loss_arrays = {}
    for alpha in ALPHAS:
        arm_reports = []
        arm_curves = []
        for seed in range(5):
            directory = batch / arm(alpha) / f'seed_{seed}'
            report = validate_training(directory)
            all_reports[(alpha, seed)] = report
            arm_reports.append(report)
            config = json.loads((directory / 'effective_config.json').read_text())
            parameters = {key: value for key, value in config['official_args'].items() if key != 'seeds'}
            if canonical_config is None:
                canonical_config = parameters
            assert parameters == canonical_config
            assert report['degree_alpha'] == alpha and report['seed'] == seed
            probes = json.loads((directory / 'periodic_probes.json').read_text())
            assert [probe['pretrain_epoch'] for probe in probes] == list(range(200, 1500, 200))
            for probe in probes:
                assert probe['epochs_completed'] == 300 and not probe['test_evaluated']
                with (directory / 'probes' / f"epoch_{probe['pretrain_epoch']:04d}_history.csv").open() as handle:
                    assert len(list(csv.DictReader(handle))) == 300
            pretrained = torch.load(directory / 'pretrained_final.pt', map_location='cpu', weights_only=False)
            assert pretrained['epoch'] == 1500 and all(torch.isfinite(tensor).all() for tensor in pretrained['state_dict'].values())
            classifier = torch.load(directory / 'classifier_best_val.pt', map_location='cpu', weights_only=False)
            assert tuple(classifier['state_dict']['linear.weight'].shape) == (7, 512)
            for name, value in config['sources_sha256'].items():
                assert digest(ROOT / 'scripts' / name) == value
            for name, value in selection['inputs_sha256'][str(directory.relative_to(batch))].items():
                assert digest(directory / name) == value
            if alpha not in [0.0, selected]:
                assert not (directory / 'test_started.json').exists()
                assert not (directory / 'test_predictions.npz').exists()
                assert not (directory / 'final_test.json').exists()
            validation.append({'degree_alpha': alpha, 'seed': seed, 'best_val_accuracy': report['best_val_accuracy'],
                               'selected_probe_epoch': report['selected_epoch'], 'initial_model_sha256': report['initial_model_sha256'],
                               'runtime_seconds': report['total_seconds'], 'pretrain_seconds': report['pretrain_seconds'],
                               'periodic_probe_seconds': report['periodic_probe_seconds'], 'final_probe_seconds': report['final_probe_seconds'],
                               'peak_allocated_mib': report['peak_allocated_mib'], 'peak_reserved_mib': report['peak_reserved_mib'],
                               'result_directory': str(directory.relative_to(ROOT))})
            with (directory / 'mask_statistics.csv').open() as handle:
                counts = list(csv.DictReader(handle))
            assert len(counts) == 2708
            assert [int(row['node_id']) for row in counts] == list(range(2708))
            masked = np.array([int(row['masked_count']) for row in counts])
            assert np.array_equal(np.array([int(row['degree']) for row in counts]), degrees)
            assert np.array_equal(np.array([int(row['degree_group']) for row in counts]), groups)
            assert masked.sum() == 1500 * 1354 and np.all((masked >= 0) & (masked <= 1500))
            for group in range(3):
                chosen = groups == group
                selections = int(masked[chosen].sum())
                assert np.isclose(report['mask_groups'][group]['mask_rate_within_group'], selections / (1500 * int(chosen.sum())))
                mask_rows.append({'degree_alpha': alpha, 'seed': seed, 'degree_group': group,
                                  'nodes': int(chosen.sum()), 'selections': selections,
                                  'mean_masked_times_per_node': float(masked[chosen].mean()),
                                  'mask_rate_within_group': float(masked[chosen].mean() / 1500),
                                  'share_of_all_masked_nodes': selections / (1354 * 1500)})
            with (directory / 'training.csv').open() as handle:
                curve = [float(row['loss']) for row in csv.DictReader(handle)]
            assert len(curve) == 1500 and np.isfinite(curve).all()
            arm_curves.append(curve)
        values = np.array([r['best_val_accuracy'] for r in arm_reports])
        loss_arrays[alpha] = np.array(arm_curves)
        validation_grouped.append({'degree_alpha': alpha, 'n_seeds': 5,
                                   'validation_accuracy_mean': float(values.mean()),
                                   'validation_accuracy_sample_std': float(values.std(ddof=1)),
                                   'validation_correct_sum': sum(r['best_val_correct'] for r in arm_reports),
                                   'runtime_mean_seconds': float(np.mean([r['total_seconds'] for r in arm_reports])),
                                   'peak_allocated_mib_mean': float(np.mean([r['peak_allocated_mib'] for r in arm_reports])),
                                   'loss_last_100_mean': float(loss_arrays[alpha][:, -100:].mean())})
    expected_selected = min([r for r in validation_grouped if r['degree_alpha'] > 0], key=lambda r: (-r['validation_correct_sum'], r['degree_alpha']))['degree_alpha']
    assert expected_selected == selected
    final_tests = []
    classification = []
    for alpha in [0.0, selected]:
        for seed in range(5):
            directory = batch / arm(alpha) / f'seed_{seed}'
            test = json.loads((directory / 'final_test.json').read_text())
            assert test['test_evaluations'] == 1 and test['selection_sha256'] == selection_sha
            assert test['test_after_alpha_selection']
            assert (directory / 'test_started.json').stat().st_mtime_ns >= selection_path.stat().st_mtime_ns
            assert digest(directory / 'test_predictions.npz') == test['prediction_sha256']
            prediction = np.load(directory / 'test_predictions.npz')
            ids, target, predicted = [prediction[key] for key in ['node_id', 'target', 'prediction']]
            assert np.array_equal(ids, expected_test_ids)
            assert np.array_equal(target, raw_graph.y.numpy()[ids])
            accuracy = float(accuracy_score(target, predicted))
            micro = float(f1_score(target, predicted, average='micro'))
            assert accuracy == test['test_accuracy'] and micro == test['test_micro_f1'] == accuracy
            final_tests.append({'degree_alpha': alpha, 'seed': seed, 'test_accuracy': accuracy,
                                'test_micro_f1': micro, 'best_val_accuracy': all_reports[(alpha, seed)]['best_val_accuracy'],
                                'selected_probe_epoch': all_reports[(alpha, seed)]['selected_epoch'],
                                'test_evaluations': 1, 'selection_sha256': selection_sha})
            for group in range(3):
                chosen = groups[ids] == group
                correct = int((target[chosen] == predicted[chosen]).sum())
                count = int(chosen.sum())
                classification.append({'degree_alpha': alpha, 'seed': seed, 'degree_group': group,
                                       'test_nodes': count, 'correct': correct, 'accuracy': correct / count})
    assert len(list(batch.rglob('final_test.json'))) == 10
    paired = []
    for seed in range(5):
        control = next(r for r in final_tests if r['seed'] == seed and r['degree_alpha'] == 0)
        improved = next(r for r in final_tests if r['seed'] == seed and r['degree_alpha'] == selected)
        paired.append({'seed': seed, 'control_accuracy': control['test_accuracy'],
                       'selected_degree_alpha': selected, 'degree_accuracy': improved['test_accuracy'],
                       'difference_percentage_points': (improved['test_accuracy'] - control['test_accuracy']) * 100})
    mask_summary = []
    for alpha in ALPHAS:
        for group in range(3):
            rows = [r for r in mask_rows if r['degree_alpha'] == alpha and r['degree_group'] == group]
            rates = np.array([r['mask_rate_within_group'] for r in rows])
            mask_summary.append({'degree_alpha': alpha, 'degree_group': group, 'nodes': rows[0]['nodes'],
                                 'mask_rate_mean': float(rates.mean()), 'mask_rate_sample_std': float(rates.std(ddof=1)),
                                 'mean_masked_times_per_node': float(np.mean([r['mean_masked_times_per_node'] for r in rows]))})
    class_summary = []
    for alpha in [0.0, selected]:
        for group in range(3):
            rows = [r for r in classification if r['degree_alpha'] == alpha and r['degree_group'] == group]
            values = np.array([r['accuracy'] for r in rows])
            class_summary.append({'degree_alpha': alpha, 'degree_group': group, 'test_nodes_per_seed': rows[0]['test_nodes'],
                                  'total_correct_across_seeds': sum(r['correct'] for r in rows), 'total_evaluations': sum(r['test_nodes'] for r in rows),
                                  'accuracy_mean': float(values.mean()), 'accuracy_sample_std': float(values.std(ddof=1))})
    group_effects = []
    for group in range(3):
        differences = []
        for seed in range(5):
            control = next(r for r in classification if r['degree_alpha'] == 0 and r['seed'] == seed and r['degree_group'] == group)
            degree = next(r for r in classification if r['degree_alpha'] == selected and r['seed'] == seed and r['degree_group'] == group)
            differences.append((degree['accuracy'] - control['accuracy']) * 100)
        group_effects.append({'degree_group': group, 'paired_difference_mean_percentage_points': float(np.mean(differences)),
                              'paired_difference_sample_std_percentage_points': float(np.std(differences, ddof=1))})
    performance = []
    for alpha in [0.0, selected]:
        values = np.array([r['test_accuracy'] for r in final_tests if r['degree_alpha'] == alpha])
        performance.append({'degree_alpha': alpha, 'accuracy_mean': float(values.mean()), 'accuracy_sample_std': float(values.std(ddof=1))})
    differences = np.array([r['difference_percentage_points'] for r in paired])
    summary = {'stage': 'M2', 'status': 'passed', 'batch': str(batch.relative_to(ROOT)), 'n_formal_runs': 20,
               'selected_degree_alpha': selected, 'selection_sha256': selection_sha,
               'validation': validation_grouped, 'test_performance': performance, 'paired_records': paired,
               'paired_difference_mean_percentage_points': float(differences.mean()),
               'paired_difference_sample_std_percentage_points': float(differences.std(ddof=1)),
               'std_ddof': 1, 'mask_groups': mask_summary, 'classification_groups': class_summary,
               'degree_metadata': metadata, 'gpu': preflight['selected_gpu'], 'rng_checks': paired_rng,
               'classification_group_paired_effects': group_effects,
               'test_count': 10, 'unselected_positive_coefficients_tested': False,
               'old_milestone_files_preserved': True}
    exception_path = batch / 'degree_alpha_0.25/seed_4/test_resume_verification.json'
    if exception_path.exists():
        summary['final_test_device_exception'] = json.loads(exception_path.read_text())
    old = json.loads((batch / 'previous_milestones_sha256.json').read_text())
    assert all(digest(ROOT / name) == value for name, value in old.items())
    write_csv(output / 'validation_summary.csv', validation_grouped)
    write_csv(output / 'validation_per_seed.csv', validation)
    write_csv(output / 'final_test_summary.csv', final_tests)
    write_csv(output / 'paired_comparison.csv', paired)
    write_csv(output / 'degree_group_analysis.csv', mask_rows)
    write_csv(output / 'degree_group_mask_summary.csv', mask_summary)
    write_csv(output / 'degree_group_classification.csv', classification)
    write_csv(output / 'degree_group_classification_summary.csv', class_summary)
    write_csv(output / 'degree_group_paired_effects.csv', group_effects)
    write_csv(output / 'loss_statistics.csv', [{'degree_alpha': alpha, 'epoch': epoch + 1,
                                               'loss_mean': float(loss_arrays[alpha][:, epoch].mean()),
                                               'loss_sample_std': float(loss_arrays[alpha][:, epoch].std(ddof=1))}
                                              for alpha in ALPHAS for epoch in range(1500)])
    save_json(output / 'summary.json', summary)
    comparison = build_comparison(summary)
    if 'final_test_device_exception' in summary:
        comparison += '\n## 资源中断与最后一次推断设备\n\n全部20次训练及前9次最终测试使用物理GPU0 L40。最后degree_alpha=0.25、seed4启动前，GPU0剩余显存低于8GiB，资源检查在test_started标记和模型前向之前停止；原始test.log保留。补测入口首次缺少官方源码路径，同样在任何测试前向之前失败，日志保留为test_resume_attempt_1.log。修复入口后，在三次检查合格、显存充足的共享物理GPU2 A40完成这个尚未测试的模型，未重训、未改系数、未重复前9次评价。\n\n测试设备变更及PID/UUID证据见该run的test_device_override*.json和test_resume_verification.json。原始训练源码、模型、选择记录未变。最后一次推断硬件与其他测试不同，浮点归约差异仍是本批次限制；未为比较设备而再次运行测试。\n'
    control_loss = next(r['loss_last_100_mean'] for r in validation_grouped if r['degree_alpha'] == 0)
    degree_loss = next(r['loss_last_100_mean'] for r in validation_grouped if r['degree_alpha'] == selected)
    comparison += '\n## 实际观察与效果判断\n\n'
    comparison += f'选定方案末100轮平均SCE={degree_loss:.6f}，Baseline={control_loss:.6f}；测试配对均值变化={summary["paired_difference_mean_percentage_points"]:+.3f}个百分点。'
    if summary['paired_difference_mean_percentage_points'] <= 0:
        comparison += '本次未观察到平均分类性能提升，不能将方法称为已验证有效的改进。'
    else:
        comparison += '本次观察到平均分类分数提高，但仅有5种子，不能推断普适性或统计显著。'
    if degree_loss < control_loss and summary['paired_difference_mean_percentage_points'] <= 0:
        comparison += '虽然重建Loss更低，分类并未提高，直接展示了两个目标并不等价。'
    comparison += '\n\n各度数组的选定方案−Baseline配对差值：\n\n'
    comparison += '| 度数组 | 均值（百分点） | 样本SD（百分点） |\n|---|---|---|\n'
    for row in group_effects:
        comparison += f'| {["低","中","高"][row["degree_group"]]} | {row["paired_difference_mean_percentage_points"]:+.3f} | {row["paired_difference_sample_std_percentage_points"]:.3f} |\n'
    with (output / 'M2_COMPARISON.md').open('x') as handle:
        handle.write(comparison)
    print(json.dumps({'selected_degree_alpha': selected, 'validation': validation_grouped, 'test_performance': performance,
                      'paired_mean_pp': summary['paired_difference_mean_percentage_points'],
                      'paired_std_pp': summary['paired_difference_sample_std_percentage_points'],
                      'mask_groups': mask_summary, 'classification_groups': class_summary}, ensure_ascii=False, indent=2))

def build_comparison(summary):
    selected = summary['selected_degree_alpha']
    lines = ['# M2：Degree-aware Masking自主实验', '', f"批次：`{summary['batch']}`。四组×五种子均完整训练；只按验证集选定正向degree_alpha={selected:g}后，再测试Baseline与该方案共10个模型。", '', '## 验证集参数选择', '', '| degree_alpha | 五种子Mean Val | 样本SD（百分点） | 最后100轮平均SCE | 平均运行秒 |', '|---|---|---|---|---|']
    for row in summary['validation']:
        lines.append(f"| {row['degree_alpha']:g} | {row['validation_accuracy_mean']*100:.2f}% | {row['validation_accuracy_sample_std']*100:.3f} | {row['loss_last_100_mean']:.6f} | {row['runtime_mean_seconds']:.2f} |")
    lines += ['', '正向系数之间以五种子最佳验证正确数量总和比较，等价于平均最佳验证准确率；精确同分选较小系数。Baseline不参与正向候选排名；即使优于所有正向方案，也如实保留最佳正向方案。alpha_selection.json在任何测试事件之前已写盘并封存输入哈希。', '', '## 最终测试及配对结果', '', '| degree_alpha | Test Accuracy Mean | 样本SD（百分点） |', '|---|---|---|']
    for row in summary['test_performance']:
        lines.append(f"| {row['degree_alpha']:g} | {row['accuracy_mean']*100:.2f}% | {row['accuracy_sample_std']*100:.3f} |")
    lines += ['', '| Seed | Baseline | 选定度数方案 | 差值（百分点） |', '|---|---|---|---|']
    for row in summary['paired_records']:
        lines.append(f"| {row['seed']} | {row['control_accuracy']*100:.1f}% | {row['degree_accuracy']*100:.1f}% | {row['difference_percentage_points']:+.2f} |")
    lines += ['', f"配对差值均值{summary['paired_difference_mean_percentage_points']:+.3f}个百分点，样本标准差{summary['paired_difference_sample_std_percentage_points']:.3f}个百分点（ddof=1）。五种子不构成充分统计证据，不宣称显著性。Micro-F1与Accuracy相同，不当作独立提升证据。", '', '## 度数分组：结构决定边界', '', '排除全部2708个自环后有10556个唯一有向邻居对，双向对称、无非自环重复。度数最小1、最大168、平均3.89808、中位数3。仅按结构分布选择尽量均衡且不拆分同度数的边界：低1–2共1068节点；中3–4共942；高5–168共698。组大小不均是保留同度节点造成，未使用类别或测试结果。', '', '| degree_alpha | 低度掩码率 | 中度掩码率 | 高度掩码率 |', '|---|---|---|---|']
    for alpha in ALPHAS:
        rows = [r for r in summary['mask_groups'] if r['degree_alpha'] == alpha]
        lines.append('| ' + f'{alpha:g}' + ' | ' + ' | '.join(f"{r['mask_rate_mean']*100:.3f}%" for r in rows) + ' |')
    lines += ['', '这里的掩码率=该组节点累计被选次数/(该组节点数×1500轮)，不是该组占全部掩码节点的份额。mean_masked_times_per_node和另一份额口径均保存在CSV；总预算始终每轮1354。', '', '| 方法 | 度数组 | 测试节点数/seed | 五seed正确数合计 | 组Accuracy均值 |', '|---|---|---|---|---|']
    for row in summary['classification_groups']:
        lines.append(f"| {row['degree_alpha']:g} | {['低','中','高'][row['degree_group']]} | {row['test_nodes_per_seed']} | {row['total_correct_across_seeds']}/{row['total_evaluations']} | {row['accuracy_mean']*100:.2f}% |")
    lines += ['', '分组测试仅从已经保存的同一次最终预测计算，未再运行模型。边界在训练前已固定，分组结果不会反馈选择系数。小组样本少，不能以局部差异作充分因果结论。', '', '## 设计解释与限制', '', '1. 节点度数衡量可用邻域上下文，高度节点可能有更多冗余信息；这只是提出挑战性掩码的动机，不保证提高分类。', '2. 原始均匀randperm使所有节点具有相同边际掩码概率；M2 Baseline仍是均匀randperm，绝未改为multinomial。', '3. 正向权重(d+1)^degree_alpha使高阶节点更常被选，固定预算下低阶节点相应减少。无放回采样的边际纳入概率不等于w/sum(w)，不能把后者当作最终概率。', '4. 四组都使用独立CPU Generator(seed)实际选点，并消耗原版同设备randperm作为shadow。替换/Dropout全局调用顺序和逐轮RNG状态已核对；CPU节点选择是相对M1的明确变化，因此M2控制组重新训练，不要求与M1逐轮或最终分数一致。', '5. 初始模型权重在同seed四组完全一致；GPU scatter的浮点归约仍可能非完全确定。相同随机流不等于所有数值运算完全确定。', '6. 高阶节点更频繁被遮蔽也会削弱其向邻居传递的信息，低阶节点监督机会减少，可能有收益和损害。具体分组差异以上表为准，不事后挑选参数。', '7. SCE只度量被选节点的特征重建，而掩码分布已改变；更快/更低重建Loss不等于线性分类Accuracy提高，不同采样目标的原始Loss也不能直接当作同一难度指标。', '8. 仅Cora、5种子、3个预设正向系数，属于课程自主探索；不声称科研新颖性、普适有效或统计显著。未使用测试集挑alpha，未删除不利seed，未更改其他模型超参数。', '', '时间及显存见validation_per_seed.csv：运行时间从worker装载参数开始，含数据加载、模型初始化、训练、全部Probe和保存，不含模块导入和启动前GPU检查；不同mask分布的CPU采样及校验成本不同。', '', 'M2到此暂停，未进入M3。']
    return '\n'.join(lines) + '\n'

if __name__ == '__main__':
    main()
