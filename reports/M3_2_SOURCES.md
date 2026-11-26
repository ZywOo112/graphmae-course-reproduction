# M3.2 正文来源清单

核查日期：2026-11-16。路径以项目根目录为基准。M1原始批次为`results/raw/M1/cora/20261113T065001179740Z/`（下称B1），M2为`results/raw/M2/cora/20261114T070950207116Z/`（B2）。所有实验图直接引用现有文件，本阶段没有重绘、修改或重新生成实验统计。

## 图片

| LaTeX label | 使用文件 | 数据/来源及核查 |
|---|---|---|
| fig:graphmae | `reports/figures/graphmae_official.jpg`（来自`third_party/GraphMAE/imgs/fig.png`） | 官方锁定Commit的3658×728像素框架图；原名.png实际编码为JPEG，故仅作字节完全相同、扩展名正确的副本，不修改第三方文件或像素。已实际查看，包含Mask、Encoder、Re-mask、Decoder、特征恢复/SCE。原论文Fig.2的GraphMAE流程部分，图注明确引用，非本项目原创 |
| fig:m1-loss | `figures/M1/loss_mean_std.png` | B1各seed的1500行training.csv；均值/样本SD；现行图哈希与M1元数据一致 |
| fig:m1-accuracy | `figures/M1/accuracy_comparison.png` | B1五份result/保存预测；论文84.2%仅作参考线 |
| fig:degree-flow | `reports/figures/degree_masking_scheme.tex` | 新增可编辑TikZ流程；依据`scripts/m2_masking.py`，无虚构采样数值；1354为真实固定预算 |
| fig:m2-validation | `figures/M2/validation_comparison.png` | `results/processed/M2/validation_summary.csv`；全部20次验证训练 |
| fig:m2-paired | `figures/M2/paired_test_accuracy.png` | `results/processed/M2/paired_comparison.csv`；控制和0.25的10次最终测试 |
| fig:m2-mask | `figures/M2/degree_mask_rates.png` | `results/processed/M2/degree_group_mask_summary.csv`及B2逐节点mask_statistics.csv |
| fig:m2-groups | `figures/M2/degree_classification.png` | `results/processed/M2/degree_group_classification_summary.csv`及同一次已保存预测 |
| fig:m2-loss | `figures/M2/loss_curves.png` | B2控制/0.25的10份training.csv，五种子逐轮均值/SD |

实际PNG尺寸/哈希及路径由`M3_2_STATIC_CHECK.json`记录。实验图为原有中文300dpi图，生成源码为`scripts/plot_m1.py`、`plot_m2.py`，字体及LICENSE在`figures/M1/assets/`；本阶段没有复制、裁切或覆盖这些图。方法流程用TikZ，不必在服务器生成PDF。官方图已有足够像素，不再从论文提取或重绘；需在本地按单栏实际尺寸确认文字可读性。

## 表格与公式

| 标签/内容 | 原始依据 |
|---|---|
| tab:dimensions；掩码、投影和Re-mask公式 | `third_party/GraphMAE/graphmae/models/edcoder.py`，原论文§3.1–3.3 |
| GAT打分、归一化、聚合 | 官方`graphmae/models/gat.py`的GAT.forward、GATConv.forward/edge_update/message；GAT论文 |
| SCE | 原论文式(2)；官方`graphmae/models/loss_func.py`；归一化含epsilon |
| 线性分类与选模 | 官方`evaluation.py`，`scripts/run_m1.py`的validation_probe；M2复用；平分取较晚轮 |
| tab:environment | `docs/ENVIRONMENT_AUDIT.md`、B1各seed环境记录；原论文附录A.3.1（`docs/paper/GraphMAE.pdf`） |
| tab:hyperparameters | 官方configs.yml解析后B1/B2逐seed effective_config.json；decoder构造固定1层 |
| tab:m1 | `results/processed/M1/summary.csv`、B1 result.json；论文对照取原文表1，不用M0结果 |
| 度数、权重、采样公式 | `scripts/m2_masking.py`；B2/degree/degree_metadata.json；不把归一化权重当最终边际概率 |
| tab:m2-validation、alpha选择式 | `results/processed/M2/validation_summary.csv`，B2/alpha_selection.json；0.5与1.0未测试 |
| tab:m2-paired、配对差值 | `results/processed/M2/paired_comparison.csv`、final_test_summary.csv；样本SD ddof=1 |
| tab:degree-groups、分组掩码/分类公式 | B2/degree/degree_metadata.json；degree_group_mask_summary.csv、degree_group_classification_summary.csv；分母区分节点数与重复评价次数 |
| tab:m2-cost | `results/processed/M2/validation_summary.csv`、validation_per_seed.csv；时间包含全部Probe，排除导入/启动资源检查/最终测试 |
| Accuracy、均值、样本SD | `scripts/analyze_m1.py`、`analyze_m2.py`；本地汇总与保存预测 |

数学符号、详细代码锚点与M0/M1/M2证据索引另见`reports/M3_PREPARATION.md`；其早期“建议重绘框架”已由本次用户优先使用官方图的要求取代。

## 文献核验

正文共实际引用7条文献，统一保存于`reports/references.bib`；不为凑条目加入未引用文献。ICLR条目采用正式会议年，URL指作者arXiv页面，页面comments明确会议归属；不把首次上传年当会议年。

| BibTeX key | 核验来源 | 版本/元数据处理 |
|---|---|---|
| hou2022graphmae | [作者arXiv页](https://arxiv.org/abs/2205.10803)、本地PDF首页和参考格式 | KDD 2022、7位作者、594–604、ACM DOI 10.1145/3534678.3539321。DOI跳转本次不可访问，以本地原文为依据；不是重新声称ACM网页访问成功 |
| kipf2017gcn | [作者arXiv页](https://arxiv.org/abs/1609.02907) | 两作者、标题、ICLR 2017；页面明确正式会议归属，2016为初次预印本年 |
| velickovic2018gat | [作者arXiv页](https://arxiv.org/abs/1710.10903) | 6作者、标题、ICLR 2018；2017是初次预印本年；保留作者变音符号 |
| velickovic2019dgi | [作者arXiv页](https://arxiv.org/abs/1809.10341) | 6作者、标题、ICLR 2019；2018是初次预印本年 |
| kipf2016vgae | [作者arXiv页](https://arxiv.org/abs/1611.07308) | 按2016预印本misc引用，note注明NIPS 2016的Bayesian Deep Learning Workshop，不冒称NIPS主会论文；doi为明确标注的arXiv DOI |
| devlin2019bert | [ACL正式页](https://aclanthology.org/N19-1423/)、[官方BibTeX](https://aclanthology.org/N19-1423.bib) | 4作者、NAACL-HLT 2019卷1、4171–4186、DOI 10.18653/v1/N19-1423均核对 |
| he2022mae | [CVF正式会议页](https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html)、[作者预印本页](https://arxiv.org/abs/2111.06377) | 正式CVPR 2022版，6作者、CVF官方BibTeX页码16000–16009；不沿用GraphMAE参考表中的2021预印本年。直接打开CVF HTML返回403，但搜索工具取得该官方页全文/BibTeX及官方PDF索引。DOI未在成功取得的一手元数据中确认，故不猜填 |

ICLR的OpenReview页面本次返回浏览器校验，没有将其算作成功读取的会议网页；作者arXiv记录及本地GraphMAE参考表相互印证。ICLR条目未填未经核实的会议DOI。正文相关研究仅概括这些成功获得的一手摘要和GraphMAE对方法的讨论，没有扩写未阅读的理论结论。

CCF资格补核：[官方数据库/数据挖掘/内容检索总目录](https://www.ccf.org.cn/Academic_Evaluation/DM_CS/)的搜索索引明确在“A类”下列出SIGKDD及其会议全称；直接打开该页和原A类子页均返回405。本次已取得**官方页面的搜索索引证据**，未获得可保存的直接网页正文，不能描述为成功下载官方目录快照。用户本地仍可人工核对。论文2022年相对本次写作2026年在五年窗口内，最终资格按实际提交日期确认。正文没有用无法访问的页面构造一条虚假文献。

## 静态检查与本地复核边界

`reports/check_sources.py`只读核对正文输入、图片、标签、引用键、花括号/环境、绝对路径及既有受保护证据哈希，写入`reports/M3_2_STATIC_CHECK.json`；不执行模型、TeX或附录生成器。现有附录入口保留，默认开关关闭，避免把旧M0生成附录当成已完成最终附件；M3.3更新后再启用。

静态检查不等于编译。最终页数、长表宽度、PNG文字缩放、TikZ的中文字体、参考文献折行与浮动体分页均需用户本地XeLaTeX实际检查。必须使用完整项目的相对目录结构，在`reports/`下编译，不能只复制main.tex。
