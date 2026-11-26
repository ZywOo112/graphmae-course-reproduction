---
name: graphmae-course-experiments
description: 在本 GraphMAE 课程项目中执行分阶段复现、度数掩码实验和 LaTeX 附录交付，维护真实实验与 Codex 记录；不适用于其他项目。
---

# GraphMAE 课程实验操作规范

先读 PROJECT_PLAN.md、PROGRESS.md 和 DECISIONS.md，确认当前授权里程碑。M2已经完成；后续须有新任务授权，不自动进入M3或增加Citeseer。

- 所有命令以项目根目录解析；使用 `.venv/bin/python`。第三方正式版本为 `third_party/GraphMAE` 的锁定 PyG Commit；审计副本是 DGL。修改前查 `git status`，不要覆盖数据或旧结果。
- M0 运行 `scripts/smoke_test.py --gpu 1`，只执行 `configs/smoke.json` 的短程预算。实际 GPU 应先确认空闲，逻辑 cuda:0 映射物理设备由脚本记录。
- 官方 `--use_cfg` 在解析 CLI 后覆盖参数；短程预算必须在加载官方配置后设置，并保存完整有效配置。官方 shell 会运行 20 个种子，不用于 M0。
- 官方 Cora 数据加载 root 依赖当前目录。将 CWD 设为项目 `data/`，不要让下载缓存污染第三方工作树。固定 public split、特征归一化和自环预处理。
- 每次实验新建时间戳目录，保存 Commit、有效配置、种子、数据 SHA256、依赖、逐轮 loss、训练/probe 时间、峰值显存及指标。Smoke 与正式结果分开标记，不把论文数字写成本机测量。
- M1/M2 用相同划分、模型和预算；以验证集选模型/α，测试集用于最终报告。保留 official periodic probe 对 RNG 的影响或对所有方法统一隔离，写入决策。
- 度数策略只改 mask 节点选择；d_i 从无自环无向图计算。固定 floor(mask_rate*N)，无放回抽样。α 不与 SCE 的 `alpha_l`（论文 γ）混淆。加权无放回的边际纳入概率不等于归一化权重。
- 阶段主要Prompt与Codex最终回答必须来自真实消息，M3统一整理；不得编造。按M1最新要求，不启动会话监听、不自动归档commentary，不以对话整理阻塞实验。
- 新增或修改自编源文件后运行 `scripts/generate_appendices.py`；它自动导入完整文件并记录哈希。第三方源码随电子附件提供，正文附录保留仓库、Commit、许可证及文件清单。
- 报告保留reports/main.tex和references.bib，由用户本地最终编译；M1不要求服务器TeX，不安装系统环境。

## M1已验证流程

使用现有venv运行scripts/preflight_m1.py先审计，只检查、不训练，输出新时间戳批次。run_m1.py读取passed的preflight，先--seeds 0；完整检查通过后--seeds 1 2 3 4。已有seed目录拒绝覆盖，重跑需新批次。

- 正式配置来自官方Cora YAML的1500/300轮，不能套smoke20轮；decoder实际核验1层。重复同值键核对解析结果。
- 历史smoke的--gpu为物理编号；M1将物理编号解析为UUID，设置CUDA_VISIBLE_DEVICES=UUID及CUDA_DEVICE_ORDER=PCI_BUS_ID，内部cuda:0，并核验PID与UUID。
- 保留官方pretrain及每200轮的7次300轮Probe，不隔离其RNG。仅运行时回调去掉测试计算；Probe按官方train更新及val>=平分规则，不接收测试标签。
- encoder固定1500轮；final Probe完整300轮后用最佳val权重。Selected Epoch从1计数，另存zero_based；选中第25轮不代表提前停止。
- 每seed独立进程，正确设Python/NumPy/Torch CPU/CUDA种子及cuDNN标志，不强制deterministic_algorithms，披露scatter非完全确定性。
- analyze_m1.py按全部5种子和ddof=1汇总，从保存预测独立复算Accuracy/Micro-F1；二者等价，不能当独立提升证据。
- 训练用项目venv；plot_m1.py用base已有Matplotlib3.8.0，不加装训练依赖。中文字体和许可证随项目保留；重绘可归档旧图，不覆盖原始实验。

## M2已验证规范

- 度数从M1同一预处理图去全部自环、按唯一邻居计数并核对双向对称，只计算一次；degree/向量、分布、分组及SHA256跨实验共享，不用标签定边界。
- 公共参数字段为degree_alpha，不与SCE alpha_l混用。四组固定0/0.25/0.5/1、5种子、1354个无重复节点，其他M1参数不变。
- m2_masking.py仅绑定encoding_mask_noise：Baseline实际CPU randperm，正向实际CPU multinomial无放回，独立Generator(seed)。四组均先消耗原设备randperm(N)作为shadow，替换沿用原randperm(k)/randperm(N)，保留取整行为。
- 正式前运行check_m2.py核对初始权重、原始替换输出及6次Dropout调用/RNG状态；四组20+20轮Pilot通过后才训练20组，验证完整1500轮paired RNG轨迹。
- 运行阶段严格分prepare/pilot/train/select/test；train和Probe不接收测试标签，select只按正向五seed平均最佳验证准确率及较小系数平分规则，封存后只测试Baseline和赢家各5个模型，其他系数不测试。
- M2控制因独立CPU选点而重跑，不假装与M1逐轮一致。所有实现保持官方源码与M1脚本不变；浮点scatter仍可能非完全确定。
- 一旦test_started或预测已存在，不重复模型评价。只有启动前资源检查失败且没有任何测试标记/预测时才可用单独补测入口，保留失败日志、选择哈希和设备变更；最后一次A40异卡推断作为限制披露。
- 汇总含全部种子、ddof=1和配对差值；分组测试只读最终已保存预测，不能反馈挑alpha。高阶掩码增多不等于分类变好；本次负结果必须保留。
- M2完成后暂停，M3只整理主要真实Prompt和最终回答，不启动监听，不把服务器LaTeX缺失设为M2阻塞。
