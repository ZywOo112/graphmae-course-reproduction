# M3 第一阶段：材料盘点、证据核查与报告结构设计

盘点日期：2026-11-14（UTC）。状态：**M0、M1、M2已完成；M3.1已完成；正文写作、对话导出、附录改造和最终交付尚未开始。**

本文件是写作准备方案，不是论文正文。项目根目录为 `/home/fuqiang/homework/机器学习课程大作业/GraphMAE/`。下文路径均相对此目录；为避免长路径反复出现，定义：

- `B0 = results/raw/M0/smoke_20261112T040413960326Z/`
- `B1 = results/raw/M1/cora/20261113T065001179740Z/`
- `B2 = results/raw/M2/cora/20261114T070950207116Z/`
- `P1 = results/processed/M1/`；`P2 = results/processed/M2/`

## 1. 材料完整性与本次核查范围

### 1.1 结论

实验主体材料完整，可进入下一阶段的正文与图表整理。论文PDF、锁定源码、依赖、Cora原始及缓存数据、M0短测、M1五次正式运行、M2二十次正式运行及十次最终测试均有实物文件。**没有发现需要补跑实验才能完成报告的数据缺口。**

尚未完成的是报告制作：参考文献目前只有GraphMAE一条；两张方法示意图需要后续设计；M1/M2真实主要对话尚未导出；附录生成脚本仍保留旧版commentary导入和M0副作用；本地XeLaTeX的最终排版尚未验证。这些不能描述为已经交付的材料。

### 1.2 实际核查及证据强度

本次读取原始文件并做离线算术核对，没有执行模型前向、训练、数据下载、已有分析/验证脚本、附录生成器或构建器。临时核查程序放在 `/tmp/graphmae_m3_inventory.py`，不属于实验实现，也不是正式复现实验入口；持久化盘点结论在本文件。

| 核查对象 | 本次实际检查 | 结论 |
|---|---|---|
| 两个官方源码目录 | Git HEAD及`status --porcelain` | PyG与DGL审计副本均清洁，Commit符合锁定记录 |
| 原论文 | 实际打开PDF、提取方法段和文献表、核对来源SHA256 | 11页，1,015,787字节，哈希一致；不是仅有下载链接 |
| 环境锁 | 用包元数据逐项对照`requirements.lock.txt` | 53条固定版本均与现有venv一致；没有安装依赖 |
| Cora | 逐个校验11个raw/processed文件；对照M0/M1数据清单 | 哈希一致；本次没有重建缓存或改变数据 |
| M1 | 5份1500行训练CSV；每份7×300行周期Probe及300行最终Probe；checkpoint哈希 | 完整、Loss有限；按最终Probe的最大验证值及较晚平分轮复核选模 |
| M1测试 | 从5份已保存NPZ预测与目标重算Accuracy，并对照逐seed及汇总记录 | 与83.98%和样本SD 1.056882个百分点一致；未重新推断 |
| M2训练与Pilot | 20份1500行训练CSV、各7×300及300行Probe；4份20+20 Pilot | 完整、Loss有限；20份官方有效训练参数一致（种子另列） |
| M2公平性证据 | 每seed四组初始权重哈希；完整1500行RNG轨迹文件逐组哈希 | 同seed四组一致；这不代表GPU浮点归约完全确定 |
| M2选择与测试 | 封存输入80项哈希；选择时间与10个测试启动时间；保存预测逐项复算 | 正向系数0.25先选择后测试；0.5和1没有测试标记、预测或最终测试JSON |
| M2度数与分组 | 度数文件SHA256、2708行逐节点掩码记录、每run总计1500×1354次选中；按组复算 | 与分组汇总一致；已保存预测的组内正确数与汇总一致 |
| 汇总与配对 | M1逐seed表，M2验证均值/样本SD、最终测试表、5项配对差值和分组表 | 未发现数值不一致；全部样本SD使用ddof=1 |
| 历史材料清单 | 独立重算`P1/artifact_manifest.json`的165项、`P2/artifact_manifest.json`的691项 | 全部匹配，无缺失或哈希差异 |
| 图片 | M1/M2现行10张PNG的文件哈希、绘图代码输入路径和元数据；抽查准确率/验证/分类分组图视觉内容 | 与清单一致；M2图对应的summary哈希一致；元数据及绘图代码均指定300dpi |
| 对话来源 | 只读定位真实Codex会话中的主要用户任务与assistant final消息边界 | M0/M1/M2来源可定位，本阶段未导出、补写或生成对话 |

范围说明：文件哈希证明当前文件与阶段封存一致，CSV和预测复算验证所报告数字；没有重演训练，也不能据此证明所有运行时细节完全确定。本次未把历史`final_verification.json`的`M3_started:false`改成当前状态，它是M2结束时的证据快照。

## 2. M0/M1/M2材料索引与可用于报告的事实

### 2.1 M0：论文、实现、环境与复现基础

| 材料 | 实际路径 | 用于报告的位置及注意事项 |
|---|---|---|
| 原论文、文本和来源 | `docs/paper/GraphMAE.pdf`、`GraphMAE.txt`、`provenance.json` | §2–3及参考文献；以PDF为准，文本抽取有公式排版损失 |
| 阅读笔记 | `docs/paper/PAPER_NOTES.md` | 方法理解索引；末尾仍是M0“尚未实现”的早期设想，不能用来描述实际M2矩阵或结果 |
| 官方实现对照 | `docs/SOURCE_AUDIT.md`、`results/raw/M0/{pyg,main}_tree.txt` | §4实现来源与附录；DGL只静态审计，不宣称两个框架数值等价 |
| 环境审计及原始输出 | `docs/ENVIRONMENT_AUDIT.md`、`results/raw/M0/server.txt`、`environment_details.json`、`dependency_check.txt`、安装日志 | §4环境表；早期GPU空闲和磁盘余量是当时快照，不当作当前资源状态 |
| 固定依赖 | `configs/requirements.in`、`requirements.lock.txt`；`scripts/setup_env.py` | 复现说明、代码附录及电子附件；训练venv与绘图base环境分开说明 |
| 数据及预处理 | `data/cora/raw/`、`data/cora/processed/`；官方`graphmae/datasets/data_util.py`；`B0/summary.json` | §4数据协议、哈希；8个原始文件、3个缓存文件 |
| 成功短测 | `B0/effective_config.json`、`epochs.json`、`summary.json`；`results/raw/M0/validation_20261112T040413067248Z/smoke_console.txt` | 只证明环境、前反向和流程可用；20+20轮结果不纳入正式复现表 |
| 历史失败与检查 | `results/raw/M0/smoke_console.txt`、`verification.json`、`final_check.json`及validation子目录 | 保留真实GPU拒绝启动、PDF提取工具缺失等历史，不删除后只展示成功 |

论文为arXiv `2205.10803v3`（2022-07-13），KDD 2022，594–604页，DOI `10.1145/3534678.3539321`。PDF SHA256为 `f297cb1597b086de05b46699854e0665a7dc74bfc1f6559b868a8ae6ac3a6da2`。

正式源码为 `third_party/GraphMAE/`，官方仓库 `https://github.com/THUDM/GraphMAE` 的PyG分支，Commit **`9127f25e01aa5ba1cca68b746284aeb7ab2b4850`**。`third_party/GraphMAE-main-audit/`是main/DGL审计工作树，Commit **`b14f080c919257b495e3cb6474286d5252d6a635`**；`third_party/GraphMAE.bundle`提供离线Git历史。两目录均保留MIT LICENSE和完整文件。

环境表可直接使用：Ubuntu 24.04.4、Xeon Gold 5320双路/104逻辑CPU、约1TiB RAM；Python 3.11.7、PyTorch 2.4.1+cu121、PyG 2.3.1、NumPy 1.26.4、torch-scatter 2.1.2+pt24cu121、torch-sparse 0.6.18+pt24cu121。驱动595.91.07、系统Toolkit 13.2.51与PyTorch实际CUDA运行库12.1分别列明。M1使用A40，M2训练使用L40；绘图复用base的Matplotlib 3.8.0，不声称matplotlib在训练锁文件内。

Cora实际为2708节点、1433维特征、7类，public split为140/500/1000。使用NormalizeFeatures，去原自环后逐节点加一个自环；10556个非自环有向条目双向对称，即5278个唯一无向邻居对，加2708自环后13264条。论文附录表列5429条边，**不能把5429、5278和13264混写为同一统计口径**；本次没有追溯论文原始计数形成原因。预训练采用全图结构与特征的transductive设置，标签仅在线性评估阶段按train/val/test用途使用。

### 2.2 M1：五种子正式复现

入口为 `M1_ACCEPTANCE.md`，但正文数值追溯到以下文件：

| 证据 | 路径 |
|---|---|
| 批次审计和统一配置 | `B1/preflight.json`、`effective_config.json` |
| 逐seed实际配置、数据SHA、软件/GPU及运行器哈希 | `B1/seed_{0..4}/effective_config.json` |
| 1500轮Loss/LR | `B1/seed_{0..4}/training.csv` |
| 周期Probe | `B1/seed_{0..4}/probes/epoch_{0200,0400,0600,0800,1000,1200,1400}_history.csv`及相应JSON、`periodic_probes.json` |
| 最终Probe及选模 | `B1/seed_{0..4}/final_probe_history.csv`、`selection.json` |
| 最终测试、时间、显存 | `B1/seed_{0..4}/result.json`、`test_predictions.npz`、`gpu_monitor.json` |
| 模型和划分 | 同目录`pretrained_final.pt`、`classifier_best_val.pt`、`splits.pt` |
| 汇总和核验 | `P1/summary.csv`、`summary.json`、`comparison.md`、`loss_statistics.csv`、`quality_checks.json`、`final_verification.json`、`artifact_manifest.json` |
| 完整日志和评估适配检查 | `B1/`的各seed控制台日志；`results/raw/M1/protocol_check_20261113T065352116738Z.json`（18节点玩具检查，不是Cora结果） |

| Seed | 最佳验证Accuracy | 选中分类器轮次（从1计） | 测试Accuracy＝Micro-F1 | worker总时长/s |
|---|---:|---:|---:|---:|
| 0 | 79.8% | 300 | 84.4% | 27.082 |
| 1 | 79.6% | 300 | 84.5% | 26.497 |
| 2 | 80.4% | 193 | 83.9% | 25.738 |
| 3 | 80.6% | 153 | 84.9% | 25.312 |
| 4 | 80.0% | 25 | 82.2% | 25.832 |

五次均完整训练1500轮、最终分类器300轮；选择第25轮不代表提前停止。固定第1500轮encoder，不用周期Probe选encoder。验证集按`>=`选择较晚平分轮，选定权重写盘后测试一次。最终 **83.98%±1.056882个百分点**（5种子、样本SD），平均worker时长26.092秒。Micro-F1与Accuracy来自同一预测且数值等价，不作为两项独立提升证据。

论文§4.1表1为 **84.2%±0.4%（20次初始化）**。M1比论文均值低0.22个百分点，仅作参考性对照，不能称为完全一致的复现；初始化次数、统计口径、DGL/PyG、软件和GPU均需披露。官方代码`np.std`默认ddof=0，本项目明确ddof=1；不能推断论文未明示的方差实现细节。

M1的5张现行图片全部位于 `figures/M1/`，详见第5节。`figures/M1/previous_render_20261113T070139533459Z/`是历史排版渲染，保留但不用于报告，避免误引旧版裁切图片。

### 2.3 M2：方法实现、验证选择与完整负结果

| 证据 | 路径 |
|---|---|
| 验收及比较分析 | `M2_ACCEPTANCE.md`、`P2/M2_COMPARISON.md` |
| 批次输入与公平性检查 | `B2/preflight.json`、`masking_checks.json`、`formal_training_check.json`、`previous_milestones_sha256.json` |
| Pilot（不进入正式统计） | `B2/pilot/{control,degree_alpha_0.25,degree_alpha_0.5,degree_alpha_1}/seed_0/`；`B2/pilot_check.json` |
| 一次性度数与分组 | `B2/degree/degrees.npy`、`groups.npy`、`nodes.csv`、`degree_distribution.csv`、`degree_metadata.json` |
| 四组正式数据 | `B2/{control,degree_alpha_0.25,degree_alpha_0.5,degree_alpha_1}/seed_{0..4}/` |
| 每次运行内容 | 上述目录内`effective_config.json`、`training.csv`、`rng_trace.csv`、7个周期Probe history、`probe_history.csv`、`classifier_selection.json`、`result.json`、`mask_statistics.csv`、`mask_group_statistics.json`、两个checkpoint、`gpu_monitor.json`、`run.log` |
| 正向系数选择屏障 | `B2/alpha_selection.json`，2026-11-14T07:27:47.306577Z先封存，含20次运行的结果、选模和checkpoint哈希 |
| 仅两组最终测试 | `B2/{control,degree_alpha_0.25}/seed_{0..4}/{test_started.json,final_test.json,test_predictions.npz,test.log}` |
| 全部验证及配对统计 | `P2/validation_per_seed.csv`、`validation_summary.csv`、`final_test_summary.csv`、`paired_comparison.csv`、`summary.json` |
| 度数组分析 | `P2/degree_group_analysis.csv`、`degree_group_mask_summary.csv`、`degree_group_classification.csv`、`degree_group_classification_summary.csv`、`degree_group_paired_effects.csv` |
| Loss及最终核验 | `P2/loss_statistics.csv`、`final_verification.json`、`artifact_manifest.json` |
| 实现与技术决策 | `scripts/m2_masking.py`、`run_m2.py`、`check_m2.py`；`DECISIONS.md` D23–D33 |

正式目录中强度数组名实际为 **`degree_alpha_1/`**，不是`degree_alpha_1.0/`。M2 `result.json`记录的是验证训练结束时`test_evaluations=0`，最终测试另存`final_test.json`；两者是阶段性记录，不能误报为结果冲突。

| degree_alpha | 验证均值 | 验证样本SD/百分点 | 测试均值 | 测试样本SD/百分点 | 用途 |
|---|---:|---:|---:|---:|---|
| 0（M2 Baseline） | 80.04% | 0.219089 | 84.20% | 0.367423 | 同协议控制组 |
| 0.25 | 79.60% | 0.761577 | 82.92% | 1.666433 | 验证选出的最佳正向方案 |
| 0.5 | 78.52% | 0.481664 | 未测试 | 不适用 | 保留完整验证训练 |
| 1.0 | 76.16% | 2.206354 | 未测试 | 不适用 | 保留完整验证训练 |

| Seed | M2 Baseline测试 | 0.25测试 | 配对差值/百分点 |
|---|---:|---:|---:|
| 0 | 84.2% | 80.1% | -4.1 |
| 1 | 84.8% | 83.5% | -1.3 |
| 2 | 84.1% | 84.0% | -0.1 |
| 3 | 84.1% | 84.2% | +0.1 |
| 4 | 83.8% | 82.8% | -1.0 |

平均配对差值 **-1.28个百分点**，样本SD **1.682855个百分点**。0.25仅在三个预设正向系数中验证表现最佳，仍低于Baseline；不能写成性能提升。0.5和1没有最终测试，不得插值、套用验证值或补造测试数字。

度数不使用标签：去2708自环后按唯一邻居计数，10556个条目双向对称、无重复；最小1、最大168、平均3.898080、中位数3。`degrees.npy` SHA256为 `26e67be7b5070414744e887427ab1188281468febdef362e57826094887321b3`；完整向量和分布已保存。分组在不拆开同度数节点的条件下，最小化三组人数与N/3的平方偏差，因此并非严格等人数三分位。

| 度数组 | 全图节点数 | 平均度数 | 每seed测试节点数 | Baseline组内掩码率 | 0.25组内掩码率 | Baseline测试Accuracy | 0.25测试Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| 低：1–2 | 1068 | 1.545880 | 400 | 50.00035% | 45.68447% | 78.75% | 77.80% |
| 中：3–4 | 942 | 3.412951 | 348 | 49.97673% | 50.52265% | 88.67816% | 87.29885% |
| 高：5–168 | 698 | 8.151862 | 252 | 50.03087% | 55.89778% | 86.66667% | 85.00% |

组内测试正确数跨5种子合计分别为Baseline **1575/2000、1543/1740、1092/1260**，0.25为 **1556/2000、1519/1740、1071/1260**。分母是重复评价的次数；每seed使用同一批400/348/252个测试节点，**不是五倍独立节点样本**。三组平均均下降，高度数组更常被掩码没有转化为分类收益。四组高度数掩码率随系数依次约50.03%、55.90%、61.67%、71.98%，说明采样分布改变确实发生。

最后100轮SCE跨seed均值依次为0.402125、0.414893、0.425383、0.442083；本次没有“Loss更低但Accuracy下降”的实证，不能为套用解释而虚构Loss改善。不同策略的损失是在不同采样目标上计算，不宜把其绝对值直接当泛化表现。

重要协议与限制：

- 同seed四组模型初始化一致；实际选点使用独立CPU Generator，Baseline为randperm，正向方案为multinomial无放回；四组先共同消耗一次原设备randperm，再沿用官方替换与Dropout顺序。详见第4节E8和第8节。
- M2 Baseline的实际选点随机流与M1不同，重新运行控制组是必要设计；84.20%与83.98%的差不能解释为方法改进效果。
- 全部20次训练及前9次测试用物理GPU0 L40；最后0.25/seed4在测试前资源检查失败，随后补测入口的源码搜索路径错误也发生在任何测试前向前。修正后仅对尚未测试的该模型在共享物理GPU2 A40推断一次。
- 该异常的实际证据在 `B2/degree_alpha_0.25/seed_4/test.log`、`test_resume_attempt_1.log`、`test_device_override.json`、`test_device_override_attempt2.json`、`test_resume_verification.json`；没有重选系数或重复前9次测试。硬件差异必须披露。
- M1总时长包括导入；M2 worker计时不包括模块导入和启动前资源检查，不能据不同批次总时长推断加速。M2内部四组均值约26.993/28.219/27.538/29.380秒，峰值张量allocated约473MiB；详见`validation_per_seed.csv`，与整卡占用区分。

### 2.4 实际代码与目录分类

本项目自编Python共 **21个文件、2803行**（按当前文本行计，不含第三方、缓存和本次临时盘点脚本）。现有附录清单收录21个脚本、3个配置、`main.tex`与`references.bib`，共26个源码/配置文件。没有根目录`models/`；模型在官方包内，这不是缺失。

| 类别 | 实际文件/目录 | 归属、修改方式及附件处理 |
|---|---|---|
| 官方原始模型、训练和数据代码 | `third_party/GraphMAE/graphmae/`、`main_transductive.py`、`configs.yml`等 | 未修改，锁定PyG；整个目录电子提交，附录列来源与目录 |
| DGL审计副本与历史 | `third_party/GraphMAE-main-audit/`、`GraphMAE.bundle` | 未修改，仅审计；不称为第二套已运行复现 |
| M0环境与最小运行 | `scripts/bootstrap_m0.py`、`setup_env.py`、`smoke_test.py` | 自编；initializer含本次历史路径，不能在已有项目直接重跑 |
| M0核验辅助 | `scripts/validate_m0.py`、`final_check.py` | 自编历史检查，可能写状态/触发旧归档；本阶段未执行 |
| M1预检和运行适配 | `scripts/preflight_m1.py`、`run_m1.py` | 复用官方pretrain，运行时替换评估回调；train/val-only Probe，不改磁盘第三方源码 |
| M1协议、统计、绘图、验收 | `scripts/check_m1_protocol.py`、`analyze_m1.py`、`plot_m1.py`、`verify_m1.py` | 自编实验检查/分析工具；完整打印，绘图依赖另说明 |
| M2掩码及实验入口 | `scripts/m2_masking.py`、`run_m2.py` | 实例绑定`encoding_mask_noise`；只调整节点选择；复用M1验证Probe及官方模型，不是重写GraphMAE |
| M2检查、分析、绘图、验收、补测 | `scripts/check_m2.py`、`analyze_m2.py`、`plot_m2.py`、`verify_m2.py`、`resume_m2_test.py` | 自编；补测脚本和失败日志必须保留，不能在提交时隐藏 |
| 报告与对话辅助 | `scripts/export_agent_logs.py`、`generate_appendices.py`、`build_report.py` | 现有机制需按第7节调整后再用；本阶段只读 |
| 配置 | `configs/requirements.in`、`requirements.lock.txt`、`smoke.json` | 3个完整自编文件；正式有效配置另在各run内保存，不另造一份替代实际值 |
| 数据、结果、图表 | `data/`、`results/raw/`、`results/processed/`、`figures/M1/`、`figures/M2/` | 数据/产物，不归作模型源码；完整电子提供，正文只选相关证据 |
| 规范和管理 | `SKILL.md`、`AGENT_WORKFLOW.md`、`README.md`、`PROJECT_PLAN.md`、`PROGRESS.md`、`DECISIONS.md` | 形成独立规范附录及电子文件；旧阶段文字按时间理解，以最新进度为准 |
| 报告现状 | `reports/main.tex`、`references.bib`、`appendices/generated.tex`、`manifest.json`、`build_status.json` | M0骨架与派生产物，未形成最终论文；本阶段保持原样 |

## 3. 最终报告目录与逐章写作计划

建议题目：**《GraphMAE的部分复现与度数感知掩码实验分析》**。主线为“理解方法—验证复现—提出假设—公平比较—解释负结果”。避免用“性能提升”“有效新算法”等未被本次证据支持的标题。

采用现有`reports/`作为报告目录，后续保留`main.tex`、`references.bib`、`appendices/`；是否拆出章节文件届时按维护需要决定，不为形式创建新目录。中文ctexart、A4单栏、可读正文、数字顺序引用；公式/图/表编号统一。

| 章节与建议小节 | 本章回答的问题及必要内容 | 公式 | 数据与图片/新增表 | 原始来源 |
|---|---|---|---|---|
| 摘要、关键词 | 做了什么、复现范围多大、度数假设结果如何；最后再写，明确负结果和5种子 | 不堆公式 | M1均值/SD、M2配对均值；不插图 | `P1/summary.json`、`P2/summary.json` |
| 1 引言 | 为什么做图自监督和特征重建？课程工作包括哪几项？区分原论文贡献与课程的实现/实验贡献 | 引用E1，通常不展开 | 不新增背景统计图 | 原论文§1；项目验收文件 |
| 2 研究背景与相关工作（图表示、生成/对比、自编码与掩码） | 监督标签不足、邻域消息传递、结构重建和特征重建的区别；用少量相关文献引出GraphMAE，不罗列无关模型 | 可引用E1、E4，详细模型放§3 | 可用简短方法对照表：目标/增强/解码器；只列文献明确支持的事实 | 原论文§2、参考文献[4][12][20][21][39][40]；待核验原始文献 |
| 3 GraphMAE算法原理与数学模型 | 3.1任务与符号；3.2为什么重建特征、掩码如何避免平凡映射；3.3多头GAT编码器；3.4线性映射、同集合re-mask和单层解码；3.5 SCE及指数作用；3.6预训练与冻结线性评估 | E1–E7逐项解释维度、集合和参数 | F1框架图；T1符号/维度；可加短算法伪代码，标注具体Cora路径而非宣称所有任务同配置 | 原论文§3/Fig.2/式(2)；官方`edcoder.py`、`gat.py`、`loss_func.py`、`evaluation.py` |
| 4 实验环境与复现方案 | 来源Commit、PyG选择、硬件软件、Cora处理/划分、transductive、固定配置、7次周期Probe、验证选模、随机性及统计定义；M0只是可运行证明 | E1、E7、E10；参数γ与degree_alpha区别 | T2环境/配置；T3数据及协议；无新增实验图 | M0审计、`B1/preflight.json`及逐seed配置；原论文§4.1/A.3 |
| 5 原始GraphMAE复现实验 | 五次完整运行是否得到接近论文的结果？逐seedAccuracy/验证轮次/时间；解释方差与预算差异 | E10 | F2/F3 Loss二选一或合并、F4 Accuracy；T4逐seed与论文分列；F5/F6可放补充材料 | `B1/seed_*/`、`P1/`、现行M1图 |
| 6 Degree-aware Masking方法设计 | 为什么考虑高度数上下文？均匀采样特点及可能损害；唯一邻居度数、权重、无放回预算、四组矩阵；CPU独立RNG及shadow调用；不更改网络或损失 | E8、E9，并引用E2/E6 | F7方法示意；T5四组矩阵；T6 M1/M2随机流差异及公平性核查 | `scripts/m2_masking.py`、`run_m2.py`、`check_m2.py`；`B2/degree/`、`masking_checks.json`、D23–D29 |
| 7 自主实验结果与分析 | 7.1四组验证与正向选参；7.2封存后的配对测试；7.3真实掩码分布；7.4各度数组分类；7.5 Loss/时长/显存；明确没有提升 | E9–E11 | F8–F12；T7四组验证及“未测试”；T8逐seed配对；T9分组人数/正确数；时间显存小表 | `B2/alpha_selection.json`及20次raw、10份预测；`P2/`全部相关CSV |
| 8 讨论与局限性 | 为什么采样改变不保证分类改善？可见邻域、目标重加权等仅为解释假设；Cora/5种子、有限系数、组不均、GPU非确定性、最后一次A40推断、M1/M2随机协议与计时差异 | 引用E6/E8/E10，不另造因果模型 | 协议差异表可复用T6；不生成无数据的机制图/显著性图 | D26、D31–D32，失败及补测记录；原论文均匀采样动机 |
| 9 总结 | 回答复现目标和自主假设；区分实现成功与性能结论，说明课程级范围 | 无新公式 | 仅引用已有结论，不新增实验建议结果 | 两阶段验收及本文件 |
| 参考文献 | 按首次引用顺序编号，作者/年份/会议/DOI准确 | 无 | 使用`unsrtnat`等与现有natbib相容方案 | `reports/references.bib`及第6节补充计划 |
| 附录A–D | 完整自编代码配置、真实主要对话、Skill工作流、第三方和证据清单 | 保留源码中的全部内容 | 全文件自动导入与电子清单，不把全部日志堆正文 | 见第7节 |

写作控制：算法章应有完整信息流、GAT计算、SCE公式和训练/评估流程；结果章必须保留每个种子及失败/限制。正文不需要把五张M1图全部重复展开，其他图及明细可放补充材料。暂不承诺页数；代码已有2803行，完整附录本身会占较多页，不能靠截断压缩。

## 4. 公式清单与依据

以下是待排入正文的公式规范，编号E1–E11仅用于本规划，不冒充原论文公式编号。矩阵按行存储节点表示。原论文与实际代码有差异时分开说明。

### E1 图、特征、划分与预处理（§3.1、§4）

\[
G=(V,E),\quad N=|V|,\quad A\in\{0,1\}^{N\times N},\quad X\in\mathbb R^{N\times F},
\qquad \bar A=\operatorname{RemoveSelfLoops}(A)+I.
\]

说明`train/val/test`互不交叠且不覆盖全体节点；特征逐行NormalizeFeatures。本次非负Cora特征可写为非零行除以行和，不能推广成任意带负特征的通用公式。依据：论文§3.1–3.2；官方`graphmae/datasets/data_util.py:45`起及`B1/preflight.json`。

### E2 均匀掩码预算（§3.2）

\[
k=\lfloor rN\rfloor,\quad M\subset V,\quad |M|=k,\quad
M\sim\operatorname{UniformWithoutReplacement}(V,k),\quad
\Pr(i\in M)=k/N.
\]

Cora为`r=0.5,k=1354`。掩码的是节点特征，节点和边仍在图中。依据：论文§3.2 Q2；`edcoder.py:168–195`的randperm。

### E3 实际输入扰动（§3.2）

\[
\widetilde x_i=\begin{cases}
x_{\rm mask},&i\in T,\\
x_{j(i)},&i\in R,\\
x_i,&\text{其他},
\end{cases}
\quad T,R\subset M,\quad T\cap R=\varnothing.
\]

token可学习、初始为零。令`q=replace_rate=0.05`，源码分别取`|T|=floor((1-q)k)=1286`、`|R|=floor(qk)=67`；故另有1个属于M的节点输入保持原特征，它仍参加re-mask和SCE。不是主动加入一套新的“保留原值”策略。依据：同上源码和`docs/SOURCE_AUDIT.md`第4项；论文Q2的理想掩码及随机替换讨论另述。

### E4 多头GAT消息传递（§3.3）

\[
e_{ij}^{(l,h)}=\operatorname{LeakyReLU}\!\left[
(a_{\rm dst}^{(l,h)})^\top W^{(l,h)}h_i^{(l)}+
(a_{\rm src}^{(l,h)})^\top W^{(l,h)}h_j^{(l)}\right],\quad
a_{ij}^{(l,h)}=\frac{\exp e_{ij}^{(l,h)}}{\sum_{u\in\mathcal N(i)\cup\{i\}}\exp e_{iu}^{(l,h)}}.
\]

\[
h_i^{(l+1)}=\mathop{\Vert}_{h=1}^{K}\sigma\!\left(
\sum_{j\in\mathcal N(i)\cup\{i\}}a_{ij}^{(l,h)}W^{(l,h)}h_j^{(l)}+b^{(l,h)}\right).
\]

这是训练时省略Dropout算子的基本计算式；正文再说明输入特征Dropout0.2、注意力Dropout0.1及PReLU。encoder两层、4头×128＝512，decoder一层1个输出头到1433维，最后解码层不套encoder的激活规则。依据：`graphmae/models/gat.py`的`GAT.forward`、`GATConv.forward/edge_update/message`；`edcoder.py`的构造；GAT原始论文[39]需补正式引用。不能把注意力系数与度数采样权重混同。

### E5 编码、投影、重掩码及解码（§3.3–3.4）

\[
H=f_E(\bar A,\widetilde X),\quad R=HW_{ED},\qquad
\widetilde r_i=\begin{cases}0,&i\in M,\\r_i,&i\notin M,\end{cases}
\qquad Z=f_D(\bar A,\widetilde R).
\]

`H,R`维度为2708×512，`Z`为2708×1433；投影无bias。论文描述decoder mask token，锁定实现明确置零。依据：论文§3.2 Q3/Fig.2；`edcoder.py:203–231`。re-mask不能解释为删除所有来自masked节点的间接信息。

### E6 SCE与优化目标（§3.5）

\[
\mathcal L_{\rm SCE}=\frac1{|M|}\sum_{i\in M}
\left(1-\frac{x_i^\top z_i}{\|x_i\|_2\|z_i\|_2}\right)^\gamma,\qquad \gamma\ge1.
\]

实际`F.normalize`有epsilon保护，可将分母范数分别写为`max(norm,epsilon)`；源码仅对M求均值，不包含边重建项。Cora的`gamma=alpha_l=3`，不是`degree_alpha`。解释余弦的范数不敏感及幂次对易例相对贡献的抑制，不宣称任意误差都缩小。依据：原论文**式(2)**；`graphmae/models/loss_func.py:5–14`。

### E7 冻结表示与线性Probe（§3.6、§4）

\[
U=f_E(\bar A,X),\quad P=\operatorname{softmax}(UW_c+b_c),\qquad
\mathcal L_{\rm cls}=-\frac1{|V_{\rm train}|}\sum_{i\in V_{\rm train}}\log P_{i,y_i},
\quad t^*=\max\arg\max_{t=1,\ldots,300}\operatorname{Acc}_{\rm val}(t).
\]

encoder固定1500轮，Probe冻结encoder；softmax仅为数学表达，代码CrossEntropyLoss直接接受logits。周期Probe只诊断验证，最终模型选择来自最终300轮Probe。依据：论文§3.3/4.1；官方`evaluation.py`及`scripts/run_m1.py:40–94`，M2复用此函数。

### E8 唯一邻居度数与加权无放回采样（§6）

\[
d_i=|\{j\ne i:\{i,j\}\in E_{\rm unique}\}|,\qquad
w_i=(d_i+1)^{\alpha_d},\quad
M\sim\operatorname{WeightedWithoutReplacement}(w,k),\quad |M|=1354.
\]

记`alpha_d=degree_alpha`。Baseline实际仍用均匀randperm，正向方案用CPU double权重和`torch.multinomial(..., replacement=False, generator=...)`。**不写`Pr(i∈M)=w_i/sum_j w_j`**：固定k、无放回的边际纳入概率不是此式。可解释逐次按剩余权重抽取的加权无放回思想，不臆造未审计的PyTorch内部随机算法。依据：`scripts/m2_masking.py:24–79,81–142`及`B2/degree/degree_metadata.json`。

### E9 验证集正向系数选择（§6、§7.1）

\[
\bar v_\alpha=\frac15\sum_{s=0}^{4}\max_t\operatorname{Acc}_{\rm val}^{(\alpha,s)}(t),\qquad
\alpha^*=\min\arg\max_{\alpha\in\{0.25,0.5,1\}}\bar v_\alpha.
\]

baseline不在“正向系数”的候选集中；所有正向方案不如baseline时仍按预先规则保留最佳正向。实际用验证正确数整数总和比较，平分取较小系数。依据：`scripts/run_m2.py:327–341`及`B2/alpha_selection.json`，不能反用测试集选alpha。

### E10 指标、样本SD与配对差值（§4、§5、§7）

\[
\operatorname{Acc}=\frac{\#\{\widehat y_i=y_i\}}n,\quad
\bar a=\frac15\sum_{s=0}^{4}a_s,\quad
s_a=\sqrt{\frac1{5-1}\sum_{s=0}^{4}(a_s-\bar a)^2},\qquad
\Delta_s=100\left(a_{0.25,s}-a_{0,s}\right)\;\text{个百分点}.
\]

Accuracy以0–1比例输入最后一个公式。单标签多分类、所有类别计入的Micro-F1等于Accuracy。波动带/误差条为样本标准差，不是置信区间。依据：两个分析脚本及已保存预测；配对式只用于M2内部。

### E11 分组实际掩码率与分类（§7.3–7.4）

\[
c_i=\sum_{t=1}^{1500}\mathbf1\{i\in M_t\},\qquad
\widehat r_g=\frac{\sum_{i\in V_g}c_i}{1500|V_g|},\quad
\bar c_g=\frac{\sum_{i\in V_g}c_i}{|V_g|},\quad
\operatorname{Acc}_g=\frac{\#\{i\in V_g\cap V_{\rm test}:\widehat y_i=y_i\}}{|V_g\cap V_{\rm test}|}.
\]

组内掩码率与“本组占所有掩码选择的份额”不同，后者分母为1500×1354；图表不得混用。依据：`m2_masking.py:145–161`、`analyze_m2.py`、逐节点掩码CSV与已保存预测。

## 5. 图表规划及来源

### 5.1 图片清单

下表F编号是规划引用，最终论文按首次出现编号。**本阶段不绘图、不重绘。**实证图现有版本可用；如后续因单栏字号需调整，只改派生排版、输出新路径并保留原图与真实数据。

| 图表名称 | 对应章节 | 原始文件路径/数据依据 | 是否已有 | 是否需要重新绘制 |
|---|---|---|---|---|
| F1 GraphMAE整体算法框架 | §3 | `third_party/GraphMAE/imgs/fig.png`，原论文Fig.2；`edcoder.py` | 已有官方英文框架图，已实际查看 | 建议后续重绘中文矢量图，注明据原论文及锁定代码整理；不可冒称原创框架 |
| F2 原始算法五种子Loss曲线 | §5 | `figures/M1/loss_curves.png`；`B1/seed_*/training.csv` | 是 | 否；可作为F3替代或并列子图，避免重复 |
| F3 原始算法Loss均值及波动带 | §5 | `figures/M1/loss_mean_std.png`；`P1/loss_statistics.csv` | 是 | 否；图注说明逐轮5种子样本SD |
| F4 原始算法五种子复现准确率 | §5 | `figures/M1/accuracy_comparison.png`；`P1/summary.csv` | 是 | 否；论文84.2%线仅为20次初始化参考 |
| F5 原始算法验证准确率与选中轮次 | §5补充 | `figures/M1/validation_results.png`；`P1/summary.csv` | 是 | 否；也可用T4表替代正文插图 |
| F6 原始算法运行时间 | §5补充 | `figures/M1/runtime_comparison.png`；逐seed`result.json` | 是 | 否；可用小表节省篇幅 |
| F7 Degree-aware Masking方法示意 | §6 | `scripts/m2_masking.py`、`B2/degree/degree_metadata.json`；计划放`reports/figures/degree_masking_scheme.*` | 无；计划路径尚不存在 | 是，后续画“去环去重→度数→权重→固定预算采样→原GraphMAE”流程，区分示意与实测 |
| F8 四组验证均值与样本SD | §7.1 | `figures/M2/validation_comparison.png`；`P2/validation_summary.csv` | 是 | 否 |
| F9 Baseline与0.25配对测试准确率 | §7.2 | `figures/M2/paired_test_accuracy.png`；`P2/paired_comparison.csv` | 是 | 否；保留全部5条线，图注提示纵轴范围 |
| F10 不同度数组实际掩码率 | §7.3 | `figures/M2/degree_mask_rates.png`；`P2/degree_group_mask_summary.csv`，原始逐节点计数 | 是 | 否；四组均展示，分母是组内节点×1500轮 |
| F11 不同度数组最终分类表现 | §7.4 | `figures/M2/degree_classification.png`；`P2/degree_group_classification_summary.csv`及保存预测 | 是 | 否；只含Baseline和0.25，400/348/252为每seed节点数 |
| F12 Baseline与0.25的Loss均值/SD | §7.5 | `figures/M2/loss_curves.png`；两组10份`training.csv`、`P2/loss_statistics.csv` | 是 | 否；不写成Loss更低的结论 |

现有10张实证图均有绘图源码：`scripts/plot_m1.py`、`plot_m2.py`；元数据在各图目录`plot_metadata.json`。中文字体`figures/M1/assets/NotoSansCJKsc-Regular.otf`及`LICENSE`也由M2复用；移动/打包时一起保留。官方`imgs/compare.png`与`ablation.jpg`属于原论文/官方结果，不作为本项目新增实证图。

### 5.2 需要编排的表格

| 表 | 内容 | 来源 | 后续处理 |
|---|---|---|---|
| T1 | 符号、矩阵维度、encoder/decoder各层头数和维度 | E1–E7、官方模型构造、有效配置 | 新排版，不新增数据 |
| T2 | 实验硬件、软件、依赖及来源Commit | M0审计、M1/M2逐run配置 | 区分运行/绘图环境及推断例外 |
| T3 | Cora规模、边统计、划分及统一训练参数 | 数据哈希清单、`B2/degree/degree_metadata.json`、官方有效配置 | 标明无向边/有向条目/自环口径 |
| T4 | M1五seed的验证/选中轮次/Accuracy/时间，另列论文参考行 | `P1/summary.csv`；论文表1 | 必须保留82.2%的seed4，论文20次与本地5次分列 |
| T5 | M2四系数×五种子，1500+7×300+300预算 | `B2/preflight.json`、各run配置 | 列清alpha_d和SCE gamma为独立参数 |
| T6 | 官方/M1/M2的评估和随机流协议差异 | 源码、D15–D18/D26–D29、`masking_checks.json` | 披露差异，不把M1和M2混成一次改进前后实验 |
| T7 | 四组验证均值/SD及最终测试适用性 | `P2/validation_summary.csv`、`final_test_summary.csv` | 0.5/1测试列写“未测试”，不用0或虚线暗示分数 |
| T8 | M2五seed配对、均值/样本SD | `P2/paired_comparison.csv` | 差值单位为百分点，含正负全部结果 |
| T9 | 分组边界/全图人数/平均度/测试人数/正确数/Accuracy | `B2/degree/degree_metadata.json`及P2分组CSV | 不将重复测试计数当独立节点数 |
| T10 | M2组内运行时间、峰值张量显存、最后100轮Loss | `P2/validation_per_seed.csv`、`validation_summary.csv` | 小表即可，明确计时范围及不同采样目标 |

## 6. 参考文献准备情况

`reports/references.bib`实际只有`hou2022graphmae`一项，作者、KDD会议、594–604页、2022年、DOI与本地PDF首页一致。其余文献尚未形成可直接使用的BibTeX；不能把“论文参考文献里出现过”写成“已经核验完整元数据”。

| 建议引用 | 用途 | 当前可定位的来源 | 下一阶段动作 |
|---|---|---|---|
| GraphMAE | 主论文、算法和原始结果 | 本地PDF、provenance、现有BibTeX | 保留，引用§3/表1/A.3时核对原文 |
| Graph Attention Networks | E4及GAT骨干 | GraphMAE文献[39]、锁定`gat.py` | 从作者原论文/正式会议页面核验作者、ICLR年份及链接后补BibTeX |
| Semi-Supervised Classification with Graph Convolutional Networks | 图卷积和Cora基准背景 | GraphMAE文献[21] | 核验原始论文元数据；不暗示本项目跑过GCN对照 |
| Variational Graph Auto-Encoders | 结构重建与GraphMAE区别 | GraphMAE文献[20] | 按真实arXiv/发表形式引用，不能虚构会议归属 |
| BERT | 掩码、预训练与推断差异 | GraphMAE文献[4] | 从ACL Anthology等一手来源核验 |
| Masked Autoencoders Are Scalable Vision Learners | 图像MAE与图MAE的设计关系 | GraphMAE文献[12]列早期arXiv版本 | 后续按所实际阅读版本核验，避免混淆预印本年和正式发表年 |
| 一种图对比学习（如DGI） | 与生成式路线简短对照 | GraphMAE文献[40]或§2 | 只选正文确实讨论的条目，核实原论文；不扩充无关综述 |
| 官方仓库、PyG/Planetoid来源 | 代码和数据来源 | `docs/SOURCE_AUDIT.md`、官方数据加载代码、M0下载记录 | 正文或脚注注明固定Commit与访问日期；需要参考文献条目时核验真实页面 |

CCF-A与近五年要求：M0的`docs/SOURCE_AUDIT.md`保存CCF官方目录URL并明确当时正文返回405，未保存可用网页快照。后续在写引言选题资格前补核官方目录证据，并按课程实际提交日期复核五年窗口；不能把失败访问包装成成功证明。本次不新增未经核验的参考文献。

数字引用可沿用现有`natbib[numbers,sort&compress]`＋`unsrtnat`，按首次引用顺序编号；正文只引用实际阅读并支持该论断的来源，不把本地实验数据归给论文文献。

## 7. 附录、真实对话和电子附件规划

### 7.1 现有机制的实际行为

| 脚本 | 已具备的能力 | 与本次最终要求的差距及处理计划 |
|---|---|---|
| `scripts/generate_appendices.py` | 自动扫描`scripts/**/*.py,*.sh`和全部configs，加入main.tex/bib；全文件`VerbatimInput`；生成`generated.tex`、文件清单和SHA256；第三方电子附件说明 | 仍导入`commentary.md`，有“监听器”占位文字；只扫描既定代码目录，不自动纳入未来章节/新代码目录。下一阶段改为显式完整清单＋Prompt/Response-only；本阶段未运行或修改 |
| `scripts/export_agent_logs.py` | 从指定JSONL匹配完全相同的真实Prompt；读取assistant final；已有不一致内容拒绝覆盖，多个final报错 | 会导出commentary、调用附录生成器、改写`results/raw/M0/final_check.json`；CLI只允许M加数字，不能直接处理M3_1等分期。后续须最小改造为一次性、无M0副作用的显式阶段/边界导出；不启用watch |
| `scripts/build_report.py` | 先生成附录，再检查XeLaTeX/BibTeX；可用latexmk或多遍XeLaTeX/BibTeX | 即使编译工具缺失也先写附录/状态，故本阶段未执行。后续按用户授权准备本地编译说明，服务器不编译 |

现有`reports/main.tex`宏使用`fvextra`：代码7.5pt/9pt行距、带行号；对话8.5pt/10pt；附录普通文字9pt/11pt；`breaklines=true,breakanywhere=true`。文件整体导入，不设`firstline/lastline`，不在不可分页的浮动体/minipage里包住长代码，可自然跨页。该方案原则上支持完整换行分页，**尚未经本地XeLaTeX验证**；中文等宽字体、长路径、特殊字符、行号和页面密度须后续检查。`fvextra`可继续使用，无须仅为改成listings而重写。

### 7.2 完整代码与规范附录

- 附录A：逐文件完整导入所有自编/修改源代码及配置，当前26项是起点；保留环境初始化、失败修复入口、检查、分析、绘图和报告辅助代码。未来新增的章节TeX、图示源、配置或代码须扩展清单，不允许仅凭旧26项宣称“全部”。
- 附录B：按M0、M1、M2、M3各子阶段顺序导入实际主要Prompt和对应最终回答。原文完整，不用总结或省略号替代；过程commentary、工具日志、系统/开发者消息、内部上下文、用户与ChatGPT讨论不进入该附录。
- 附录C：完整纳入`SKILL.md`、`AGENT_WORKFLOW.md`，必要的`PROJECT_PLAN.md`、`DECISIONS.md`及项目操作说明。该规划和最终附件清单电子保留；是否打印本规划按其最终必要性决定，不重复整篇报告。
- 附录D：官方仓库/分支/Commit/许可证、第三方目录清单、数据和环境来源、实验文件索引与校验清单。未修改官方代码不必整库打印，电子附件必须实际包含完整源码。当前没有修改过的第三方文件或patch，适配源码全部位于scripts内并归附录A。

生成的`appendices/generated.tex`不递归导入自身，电子附件保留其完整文件及生成器。源码附录必须由最终源码重新生成并比对清单，不人工复制代码到正文造成版本漂移。

### 7.3 真实对话来源（仅定位，未导出）

已读取的原始Codex会话源为：

`/home/fuqiang/.codex/sessions/2026/11/12/rollout-2026-11-12T03-48-51-01a1cddf-6163-7061-a1f3-bdf5774aa614.jsonl`

它在项目目录之外，不能依赖最终收件人仍能访问这个绝对路径；后续必须导出必要原文到项目内。仅提取对应实际任务，不将整份会话当附件。

| 阶段 | 主要Prompt时间（UTC） | Prompt字符数 / SHA256 | 对应最终回答时间（UTC）及现状 |
|---|---|---|---|
| M0 | 2026-11-12T03:49:27.889Z | 5173；`b9cb7c370c64fb6cb1f1e4e0db54fcd2f9287eee39fbf5b56d72289f815eca15` | 2026-11-13T05:36:35.642Z，1756字符；已有`agent_logs/M0/{prompt,response}.md`和`export_status.json` |
| M1 | 2026-11-13T06:44:18.487Z | 7394；`7aabe7111e8047c32978cd546e9bc38ce1eadbdf42956264879e42335d5b3b19` | 2026-11-13T07:13:08.334Z，1181字符；真实final可定位，尚未导出 |
| M2 | 2026-11-13T07:26:31.941Z | 7232；`1cd310a37d00de07373faa4e4659d5a24b7646fbfea892ef1a1878c74830081b` | 2026-11-14T08:37:49.381Z，1507字符；真实final可定位，尚未导出 |
| M3.1 | 2026-11-14T11:40:32.780Z | 4892；`47d64d8c64d2c5522f3c78357f5be4db15022cc6a25f005df2c37d9056287c32` | 本次任务进行中，核查时尚无最终消息；不能预写最终回答 |

M1真实final SHA256：`e34b865aaeeb1413b8c6a80d52aa4b74c028bb1245eb2711dc5e4bd45573bcd4`；M2为`7a963f79eb14314abea803e2db5a71bec1e1e67178ca3fd10a35d927044d3296`。这些是来源定位信息，不代替原始对话。字符数按所解析文本计，后续导出须保留原文换行并单独校验。

M0现有commentary文件保留为历史电子材料，本次不删除；最终对话附录不导入它。M3将有多个授权子阶段，应显式使用子阶段映射/目录（如M3_1），不能拿一个M3目录拼接成“单次完整回答”，也不能用当前摘要替换真实final。后续只做一次性导出与人工边界复核，不启动会话监听。

### 7.4 电子附件组织与可移植性

继续使用现有根目录，不复制成多份相互漂移的源码。最终电子包应包含：README/管理规范；全部scripts与configs；两份完整官方工作树及LICENSE、Git bundle；论文PDF与provenance；11个Cora文件；M0/M1/M2全部原始日志、失败日志、有效配置、checkpoint、预测和汇总；现行图片、必要历史渲染、字体与许可证；真实主要对话；报告TeX/BibTeX、附录、实际用图和图示源、本地编译后的PDF及文件SHA256清单。

`.venv/`、`__pycache__/`、下载缓存不作为源码打印，可从电子包排除；依赖锁和安装来源必须保留。版本锁没有wheel哈希且没有完整wheelhouse，不能声称依赖可完全离线安装。完整代码/数据/模型附件已存在，但“任意新机器一键复现”尚未验证。

现有两个Git工作树的`.git`元数据含机器路径，移动后可能失效；保留源码实物及bundle，在最终包装说明中给出从bundle恢复Git历史的方式，不改动当前实验仓库。附录路径是`../scripts/...`等相对引用，必须解压完整项目后在reports目录编译，不能只提供reports文件夹。绘图脚本含已记录的原始绝对路径元数据，搬移时应使用已提供的输入参数/相对批次路径，不能把历史记录改写成新机器路径。

本次实际文件体积（不含venv/pycache及Git对象库）：M1 raw约42.6MB、M2 raw约215.6MB、Cora约16.3MB、third_party文件及bundle约24.8MB、figures含字体和旧渲染约19.8MB；最终包还需计入Git对象库、后续文稿和PDF。无需为控制附件大小删除checkpoint或不利结果。

## 8. 缺口、优先级与必须披露的限制

### A. 正式写作前必须确认（本次已完成，无阻塞缺失）

| 项目 | 本次状态 | 后续约束 |
|---|---|---|
| 论文PDF、公式可靠来源、源码Commit | 已确认完整，PDF/代码与记录相符 | 公式引用用原文和锁定代码，不直接复制抽取乱码 |
| 原始实验和预测可追溯性 | 已确认5+20训练完整、10次M2最终测试有效、清单哈希无差异 | 不补跑、不覆盖；若后续发现变化须停用该证据并记录 |
| 核心结果边界 | 已固定：M1 83.98%；M2控制84.20%、0.25为82.92%；0.5/1未测试 | 正文/摘要/图表都遵循该边界 |
| 真实主要对话来源是否存在 | M0/M1/M2真实来源均定位；M3.1待当前final实际产生 | 未导出不等于丢失；导出前不得宣称对话附件已交付 |

### B. 正式写作过程中解决（不需要新实验）

| 缺口/问题 | 影响与具体动作 | 验收方式 |
|---|---|---|
| 参考文献目前仅1项；CCF目录未有成功正文证据 | 按第6节核验需要的原始文献并填BibTeX；补选题资格证据及实际截止日期五年窗口核对 | 逐条链接/元数据支持正文，不虚构DOI/会议信息 |
| 旧阅读笔记、README和计划含历史“未进入M3”/早期M2设想 | 本文件及PROGRESS为当前授权状态；写正文不用早期设想代替实际矩阵；后续统一梳理历史与现状 | 实际参数/结论与raw记录一致，旧阶段证据不追改 |
| 缺中文算法与度数策略示意图 | 后续按F1/F7制作矢量示意；注明来源和示意性质 | 含真实信息流、re-mask和固定预算，无虚构实验曲线 |
| 附录生成器与当前对话要求不一致 | 后续取消commentary导入、监听器占位和M0状态副作用；扩展完整源码发现/显式清单 | Prompt/Response逐字校验，源码全文件导入，无省略 |
| 主要对话尚未落盘；M3有多个子阶段 | 后续一次性按明确消息边界导出，记录来源/时间/哈希；当前final产生后再提取 | M0/M1/M2及每个M3子阶段原始Prompt与final一一对应 |
| 最终TeX尚未写作；中文代码/长路径分页未知 | 由后续授权写作阶段准备源码，用户本地XeLaTeX编译检查；服务器不安装/编译 | 正文、参考文献、交叉引用、附录全部完整且打印可读 |
| 附件清单和Git路径可移植性 | 最终冻结时重新生成完整清单，保留bundle、字体LICENSE、图示源、全部结果；检查解压相对路径 | 附件可定位所有正文证据；不只留下载链接 |

### C. 不影响报告完成，以限制披露

1. 只有Cora、5个配对种子和3个预设正向系数，不是原论文21数据集/20次初始化的完整复现；不做充分统计证据或普遍无效的结论。
2. 原论文DGL历史软件/2080Ti等环境与当前PyG/A40/L40不同；数据边统计口径也不直接等同，未追溯的5429与5278差异如实说明，不杜撰原因。
3. cuDNN标志和RNG轨迹已经管理，但PyG scatter等GPU归约仍可能非完全确定；一致RNG不等于位级确定性。
4. M1与M2控制组实际节点选择随机协议不同，不能用二者均值差宣称性能提升；正式算法比较只在M2内部。
5. 最后一个模型的唯一测试在A40而非L40完成；失败发生在测试前向前，日志已保留，未重选系数；承认可能的浮点硬件影响。
6. 以同一验证划分在五seed均值上选正向系数，是有限的开发集选择；测试不反馈调参，但不提供跨数据集泛化证据。
7. 相同度数保持同组导致组人数不均，测试组仅400/348/252个固定节点，五seed不构成五倍独立测试节点；组间差异仅作描述。
8. 采样策略确实改变掩码频率，但本次整体及三组平均分类均下降。低度节点监督减少、枢纽上下文受扰动、重建目标被重加权均是可能解释，未通过机制消融证实。
9. 各方案Loss来自不同被采样节点分布；本次0.25的Loss也更高，不能泛化成“重建改善但分类退化”的观察。时长不同计量边界、显存allocated与整卡占用必须区分。
10. 最终PDF/打印质量必须在本地实际编译后验收；当前没有main.pdf是阶段安排，不是假定编译成功，也不需要服务器TeX来消除。

## 9. 下一阶段执行计划与本阶段停止点

下一阶段须由用户另行授权。本次已具备进入M3第二阶段的材料条件，建议按以下连续任务组织，不再做正式实验：

1. 按第3节确定章节与表格，先补齐必要一手参考文献及选题资格证据，统一符号、百分比/百分点和M1/M2协议说明。
2. 依据现有数据编排T1–T10及现行实证图，绘制F1/F7方法示意；如需调整图片排版保存新派生版本，不改raw数据。
3. 写作中文论文正文与参考文献，突出GraphMAE完整数学信息流与M2负结果；不得让摘要把“实现成功”写成“性能提升”。
4. 在对应授权阶段最小调整附录/导出工具，一次性归档真实主要Prompt和final；完善全部自编源文件清单及规范附录；不监听会话、不追改M0证据。
5. 准备完整本地编译源码包、图表/字体/许可证、电子源码及实验证据清单，明确本地XeLaTeX/BibTeX操作；编译和打印复核由后续本地环节完成。

M3.1交付仅为本文件与`PROGRESS.md`当前状态更新。现有`reports/main.tex`、`references.bib`、`appendices/`保持原样；未撰写完整论文、未导出对话、未运行附录/构建脚本、未训练或测试模型、未安装系统依赖。**到此暂停，等待下一阶段指令。**
