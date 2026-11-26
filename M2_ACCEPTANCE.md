# M2验收：Degree-aware Masking自主实验

2026-11-14：**通过M2交付验收。算法成功实现，但本批次未观察到性能提升。完成后暂停，不进入M3。**

## 完成范围与协议

四组degree_alpha=0/0.25/0.5/1，seeds0–4，20次完整1500轮预训练、各7次300轮周期Probe和最终300轮Probe；另有四组seed0的20+20轮Pilot，不纳入正式统计。

官方PyG Commit保持9127f25e01aa5ba1cca68b746284aeb7ab2b4850，第三方和M1脚本不变。Cora SHA256、public split140/500/1000、NormalizeFeatures及自环与M1一致；512维、4heads、2层GAT encoder、1层decoder、mask0.5、replace0.05、SCE alpha_l=3、Adam/cosine及全部学习率/dropout/预算不变。

度数去2708自环后按10556个唯一邻居对计算，双向对称、无非自环重复。min1/max168/mean3.89808/median3。仅按结构分布固定低1–2/中3–4/高≥5，1068/942/698节点；向量、分布及SHA256保存在degree/，只计算一次共用。

四组实际选点均用独立CPU Generator(seed)：Baseline均匀randperm，正向multinomial无放回；每组都消耗原版同设备randperm(N)作shadow，保持后续随机替换与Dropout流。每轮均1354个不同节点。同seed初始权重、真实GPU替换/Dropout诊断和全部1500轮CPU/CUDA全局RNG轨迹核对一致，但不声称浮点scatter完全确定。相对M1移到独立CPU选点是明确的协议变化，M2基线重跑，不能复用M1分数。

## 验证选择

| degree_alpha | 验证均值 | 样本SD（百分点） |
|---|---|---|
| 0 | 80.04% | 0.22 |
| 0.25 | 79.60% | 0.76 |
| 0.5 | 78.52% | 0.48 |
| 1 | 76.16% | 2.21 |

全部正向系数的验证均值均低于Baseline。仍按预定规则在正向方案中选择0.25；五seed最佳验证正确数总和最大，精确同分选较小系数。选择记录alpha_selection.json于07:27:47 UTC封存及记录输入模型/结果哈希，先于07:28:08 UTC首个测试结果，未用测试选参。

## 最终测试

M2 Baseline=84.20%±0.37个百分点；选定degree_alpha=0.25为82.92%±1.67个百分点。配对差值均值-1.28个百分点，样本标准差1.68个百分点（全部ddof=1）。

| Seed | M2 Baseline | degree_alpha=0.25 | 配对差值（百分点） |
|---|---|---|---|
| 0 | 84.20% | 80.10% | -4.10 |
| 1 | 84.80% | 83.50% | -1.30 |
| 2 | 84.10% | 84.00% | -0.10 |
| 3 | 84.10% | 84.20% | +0.10 |
| 4 | 83.80% | 82.80% | -1.00 |

只测试Baseline和0.25的10个模型，各一次；0.5及1没有测试文件。预测已保存并独立复算，Micro-F1与Accuracy相同，不当独立证据。

## 度数组与训练过程

| 度数组 | 全图节点数 | 测试节点数/seed | Baseline组内掩码率 | 0.25组内掩码率 | Baseline分类Accuracy | 0.25分类Accuracy |
|---|---|---|---|---|---|---|
| 低：1–2 | 1068 | 400 | 50.00% | 45.68% | 78.75% | 77.80% |
| 中：3–4 | 942 | 348 | 49.98% | 50.52% | 88.68% | 87.30% |
| 高：≥5 | 698 | 252 | 50.03% | 55.90% | 86.67% | 85.00% |

每组完整正确数与每seed结果见degree_group_classification.csv。组边界未使用测试标签；分组评价只从同一次保存预测导出，未再运行模型。三组平均分别下降约0.95、1.38、1.67个百分点，高阶节点并未因更多监督而受益；局部小样本差异不作充分因果结论。

0.25低/中/高组平均被选约685.27/757.84/838.47次，Baseline约750.01/749.65/750.46次（1500轮）。实际掩码频率随degree_alpha增强而向高阶转移，说明实现改变了选点分布；归一化权重不等于无放回边际纳入概率。

末100轮平均SCE：Baseline0.402125，0.25为0.414893，0.5为0.425383，1为0.442083。本次重建Loss也未改善；即使其他实验出现更低Loss，也不等于分类提高，因为被重建目标分布和下游目标不同。Baseline与0.25平均worker计时26.99/28.22秒，显存均约472.73MiB。计时不含模块导入及启动前GPU检查，含参数装载后数据、初始化、预训练、全部Probe和保存；CPU采样/校验成本及服务器负载影响速度，不作严格benchmark结论。

## 产物

- 原始：results/raw/M2/cora/20261114T070950207116Z/，含degree/、pilot/、四个正式组及seed_0至seed_4。
- 代码：scripts/{m2_masking,run_m2,check_m2,analyze_m2,plot_m2,verify_m2,resume_m2_test}.py；复用M1验证Probe和官方模型/训练。
- 汇总：results/processed/M2/validation_summary.csv、validation_per_seed.csv、final_test_summary.csv、paired_comparison.csv、degree_group_analysis.csv、分组分类及配对CSV、loss_statistics.csv、summary.json、M2_COMPARISON.md。
- 图表：figures/M2/{validation_comparison,paired_test_accuracy,degree_mask_rates,degree_classification,loss_curves}.png，中文300dpi、保留绘图代码和已有Noto字体来源；未改M1图。
- 配置、逐轮loss/LR、完整Probe、初始权重哈希、RNG轨迹、mask统计、预训练模型和最佳分类器均保留；原始训练/验证result.json不追加测试指标，最终评价另存final_test.json。

## 问题与限制

全部训练和前9次测试在物理GPU0 L40；最后0.25/seed4启动前GPU0显存不足8GiB被资源守卫停止，没有模型测试前向。补测入口一次源码路径错误也发生在测试前向前，已修复并保留日志。最后这个尚未评价的模型在共享GPU2 A40完成唯一一次推断，设备/三次资源快照/PID/UUID记录齐全；未重训、未改系数、未重复前9次。该次异卡浮点归约差异是本批次限制，未为比较设备再跑测试。

M0/M1原始产物和官方源码SHA256保持不变。没有失败正式训练；两次失败测试启动记录未删除。没有安装依赖、后台对话监听或服务器LaTeX编译。

仅Cora、5种子、3个预设正向系数，课程级自主探索，不声称科研新颖性或普适结论。可以提出“度数偏置削弱低阶监督或邻域可见信息”的解释，但本实验不能单独证明机制；实现正确不等于算法有效。

已具备进入M3的完整材料，包括真实复现、预设改进、负结果和限制。M2到此暂停，等待M3明确任务。
