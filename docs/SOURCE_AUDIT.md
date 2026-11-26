# 论文与官方实现审计

审计时间：2026-11-12。官方仓库 https://github.com/THUDM/GraphMAE 。

## 来源与版本

| 项目 | 锁定来源 | 保存位置 |
|---|---|---|
| 论文 | arXiv 2205.10803v3，2022-07-13；KDD 2022，pp.594–604，DOI 10.1145/3534678.3539321 | docs/paper/GraphMAE.pdf，哈希见 provenance.json |
| 官方 PyG | branch pyg，9127f25e01aa5ba1cca68b746284aeb7ab2b4850 | third_party/GraphMAE |
| 官方 DGL | branch main，b14f080c919257b495e3cb6474286d5252d6a635 | third_party/GraphMAE-main-audit（同仓库 detached worktree） |

仓库含完整源码、图片、chem 子项目及其官方预训练权重，未只保留下载链接。两分支的 `git ls-tree -r` 原始目录清单保存于 results/raw/M0/{pyg,main}_tree.txt；末次状态与 Commit 见 verification.json。DGL 仅静态审计，未安装/运行 DGL，不能声称两框架数值一致。

## 按执行路径对照

| 位置 | DGL main | PyG pyg |
|---|---|---|
| README/依赖 | Python>=3.7、torch>=1.9、DGL>=0.7.2；README 指 PyYAML5.4.1 | Python>=3.7、torch>=1.9、PyG>=2.2.0；env.yml 冗长旧环境，不能视为最小依赖 |
| requirements.txt | pyyaml/tqdm/tensorboardX/sklearn/ogb，未含 torch 或图框架 | 同左，需额外安装 torch/PyG/扩展 |
| 数据 | graphmae/datasets/data_util.py，DGL CoraGraphDataset 等 | 同名文件，Planetoid + NormalizeFeatures，默认 public split，root="" |
| 预处理 | 删除/添加自环；Cora 特征由 DGL 数据集提供 | 删除/添加自环；Cora 行归一化；必须核对跨框架差异 |
| 模型 | edcoder.PreModel.forward(g,x) | edcoder.PreModel.forward(x,edge_index) |
| GNN | 自定义 DGL GAT/GCN/GIN 等 | 自定义 PyG MessagePassing GAT/GIN；不是直接替换为当前 PyG GATConv |
| 掩码 | randperm 无放回选 floor(r*N) 个节点；token/noise 替换 | 同等基本逻辑，输入接口不含图，度数扩展以后需传入结构或预计算权重 |
| re-mask | encoder_to_decoder 后将 masked 节点置零 | 同左，图 decoder 走 re-mask；MLP/linear 分支不用 |
| 损失 | loss_func.sce_loss | 同文件哈希，normalize 后 pow(alpha) 再 masked 节点均值 |
| 训练 | main_transductive.py；全批量 Adam，可 cosine；每200轮probe | 同名入口与相同整体流程 |
| 评估 | evaluation.py 冻结 embedding 后线性分类器 | 同左；train标签训练，val选最佳probe，最后返回 final_acc/estp_acc |
| 配置 | configs.yml；run_transductive.sh | 两分支 configs.yml 相同 git blob；shell 参数可被 use_cfg 覆盖 |

## Cora 正式基线参数

configs.yml: lr=0.001，lr_f=0.01，num_hidden=512，num_heads=4，num_layers=2，weight_decay=2e-4，weight_decay_f=1e-4，max_epoch=1500，max_epoch_f=300，mask_rate=0.5，encoder=gat，decoder=gat，activation=prelu，in_drop=0.2，attn_drop=0.1，linear_prob=true，loss_fn=sce，drop_edge_rate=0，optimizer=adam，replace_rate=0.05，alpha_l=3，scheduler=true。decoder 层数在构造中固定为1，不是跟 encoder 同为2层。

论文表1 Cora Accuracy=84.2±0.4%，Citeseer=73.4±0.4%；这些是论文报告值，本项目 M0 不声称复现。论文节点分类用20次初始化；课程 M1 首批5种子须明示缩减。论文 A.3 的历史环境为 PyTorch1.9/DGL0.7.2/CUDA10.2/Python3.7，当前运行环境不同。

## 实际发现及处理

1. `--use_cfg` 在 CLI 后读取 YAML 并写回 args，直接 `--max_epoch 20 --use_cfg` 会恢复1500。M0 包装器先加载官方 YAML，再覆盖并保存20+20轮。YAML 有重复同值 num_layers，safe_load 取最后值；配置冻结时需规范化。
2. 官方 shell 默认为20个种子、1500+300轮；M0 不执行此 shell。
3. 官方 set_random_seed 将 cudnn.deterministic 拼成 determinstic；不能据此保证确定性。GPU scatter 原子加法也可能不完全确定。M1需写实记录；若增强确定性需统一所有方法并测试。
4. token/noise 两次 floor：Cora1354个 masked 节点中1286个token、67个noise、1个因取整保留原特征；mask集合仍1354个并计入loss/re-mask。保持官方行为，不静默修正。replace_rate很小时 `[-0:]` 潜在切片错误，但本次Cora配置不触发。
5. PyG README 提到 inductive 脚本，但该分支没有 main_inductive.py；本项目只采用经验证的 Cora transductive 路径，不声称其全部任务可运行。
6. DGL decoder 用原图，PyG 在 drop_edge_rate>0 时复用丢边图。Cora drop_edge_rate=0，此区别不激活；不扩展宣称两个版本处处等价。
7. 官方周期probe会消耗RNG、每轮显示test指标；M1须明确其保留/隔离方案且不以test调参。官方评估选 val>=best，平分选较晚 epoch。
8. checkpoint.pt 使用固定名称，不适合多次实验；本阶段不写它，M1用每run独立路径。

## 版本选择结论

选择官方 PyG 分支的 Cora 路径，依据是接口审计、成熟轮子可安装及 GPU smoke 的实际结果（见 ENVIRONMENT_AUDIT.md 和 summary.json）。保持官方源码未修改，新增脚本只负责隔离目录、限制轮数和保存证据。选择理由不是泛称 PyG 比 DGL 更好；DGL 是忠于论文历史实现的备选，但在当前服务器上尚无运行验证。

## M1执行补充（2026-11-13）

正式五种子已完成，第三方Git仍清洁。scripts/run_m1.py直接调用官方main_transductive.pretrain，在当前进程替换其node_classification_evaluation回调；不修改磁盘源码。

适配后的周期Probe与官方同样在200至1400轮每200轮执行完整300轮；相同embedding、官方LogisticRegression、Adam、交叉熵、每轮train/val前向、>=验证选模及deepcopy规则。仅去掉test准确率/损失计算和打印；不隔离RNG。check_m1_protocol.py在独立18节点fixture上验证选中权重、轮次和CPU RNG一致，记录在results/raw/M1/protocol_check_20261113T065352116738Z.json。此测试不是Cora正式结果，也不证明GPU算子完全确定。

预训练图删除y/test_mask，Probe参数仅含训练/验证标签；最终固定encoder第1500轮，重新训练300轮分类器，写selection和best checkpoint后测试一次。最后300轮仍完整执行，验证最优在较早轮不触发提前停止。CPU模型往返及分类器CPU初始化顺序保留；入口正确设置cuDNN标志是本次明确要求的差异。

M1最终Accuracy83.98%±1.05688个百分点（5种子，ddof=1），详见M1_ACCEPTANCE.md，不与M0短测混用。以后公平对照需保持本包装器协议。

## 一手资料链接

- 论文：https://arxiv.org/abs/2205.10803v3
- DGL 官方仓库：https://github.com/THUDM/GraphMAE/tree/b14f080c919257b495e3cb6474286d5252d6a635
- PyG 官方分支：https://github.com/THUDM/GraphMAE/tree/9127f25e01aa5ba1cca68b746284aeb7ab2b4850
- PyG 安装文档：https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
- PyTorch 历史版本安装：https://pytorch.org/get-started/previous-versions/
- CCF 官方 A 类会议目录：https://www.ccf.org.cn/Academic_Evaluation/DM_CS/zgjsjxhtjgjxshy/al/ （本次网页正文返回405；搜索索引可见官方条目，保留访问限制，不据此伪造网页快照。）
