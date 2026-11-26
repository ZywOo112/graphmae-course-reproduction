# M1验收：Cora原始GraphMAE正式复现

2026-11-13：**通过，5/5正式实验完整成功。M1到此暂停，不进入M2。**

## 配置与协议

官方PyG Commit `9127f25e01aa5ba1cca68b746284aeb7ab2b4850`，第三方工作树未修改。使用原venv（Python3.11.7、torch2.4.1+cu121、PyG2.3.1），未安装/重装训练依赖。

Cora public split140/500/1000；2708节点、1433维、7类；NormalizeFeatures及每节点一个自环；原始/processed SHA256与M0一致。512维、4heads、2层GAT encoder、1层GAT decoder；mask0.5、replace0.05、SCE alpha_l=3；Adam lr0.001、wd0.0002、cosine、特征dropout0.2、注意力dropout0.1、drop_edge0；probe lr0.01、wd0.0001。

每seed完整1500轮预训练、7次周期300轮验证Probe及最终300轮线性评估。encoder固定1500轮；最终分类器按val>=选较晚平分轮。选择写盘后仅一次测试，Probe不接收测试标签。物理GPU2 A40通过UUID映射内部cuda:0；五次启动均非共享。

## 实际结果

| Seed | Best Val Accuracy | Selected Epoch（从1计数） | Test Accuracy | Micro-F1 | Runtime（秒） |
|---|---|---|---|---|---|
| 0 | 79.80% | 300 | 84.40% | 84.40% | 27.08 |
| 1 | 79.60% | 300 | 84.50% | 84.50% | 26.50 |
| 2 | 80.40% | 193 | 83.90% | 83.90% | 25.74 |
| 3 | 80.60% | 153 | 84.90% | 84.90% | 25.31 |
| 4 | 80.00% | 25 | 82.20% | 82.20% | 25.83 |

Accuracy均值83.98%，样本标准差1.05688个百分点（ddof=1）；Micro-F1相同，来自同一单标签多分类预测。平均运行26.09秒，合计130.46秒；峰值张量显存472.70MiB。时间含worker导入、初始化、训练和保存，不含共享preflight/父进程复核。

论文参考84.2±0.4%，20次初始化；本项目5种子均值低0.22个百分点、波动更大。软件框架与GPU不同，不能宣称完全相同复现或统计等价。未改参数、增加有利种子或重跑seed4。

## 产物与检查

- 原始批次：results/raw/M1/cora/20261113T065001179740Z/；含preflight、配置、GPU快照、seed_0至seed_4及独立控制台日志。
- 每seed：training.csv、probes/、final_probe_history.csv、selection.json、result.json、effective_config.json、splits.pt、test_predictions.npz、pretrained_final.pt、classifier_best_val.pt、gpu_monitor.json、status.json。
- 汇总：results/processed/M1/{summary.csv,summary.json,comparison.md,loss_statistics.csv,quality_checks.json}。
- 五张中文300dpi图：figures/M1/{loss_curves,loss_mean_std,accuracy_comparison,validation_results,runtime_comparison}.png；绘图代码plot_m1.py、字体/许可证及哈希随附。
- 代码：scripts/{preflight_m1,run_m1,check_m1_protocol,analyze_m1,plot_m1}.py；玩具用例验证适配前后Probe权重/轮次/CPU RNG一致，未混为实验结果。
- 检查通过：五次1500轮、各8×300轮Probe、有限loss/梯度、模型更新、encoder评估不变、验证选模、预测指标复算、checkpoint形状/哈希、数据SHA256与官方Git状态。

## 问题、限制与下一阶段

初查满载时未启动；随后空闲A40通过三次采样后使用。官方每轮计算test的问题已用项目回调适配，第三方不变。旧PyG的torch.load FutureWarning不影响固定环境运行。图表标题裁切只修正排版，旧渲染保留。

GPU scatter仍可能非完全确定；seed4较早验证最优和较低测试分数如实保留。5种子不足以证明稳定达到论文方差，目前无证据要求调参。M1不要求服务器LaTeX或复杂对话归档；报告由用户本地编译，M3整理主要真实Prompt与最终回答。

已具备进入M2的基线条件，建议沿用相同5种子、预算、数据及验证协议公平比较。本阶段未实施度数掩码、Citeseer或M2，等待下一阶段明确任务。
