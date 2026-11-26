# 服务器环境与运行可行性审计

时间：2026-11-12 UTC。命令输出、安装日志与包清单见 results/raw/M0/；本文件区分系统能力、当前环境和真实短测。

## 硬件与系统

| 项目 | 实测 |
|---|---|
| Linux | Ubuntu 24.04.4 LTS，x86_64 |
| CPU | 2×Intel Xeon Gold 5320 @2.20GHz；52物理核、104逻辑CPU |
| 内存 | 约1.0TiB；初查available约974GiB；swap63GiB |
| 磁盘 | /dev/sda4，3.4TiB总量，初查剩余约422GiB（88%已用） |
| GPU | 2×NVIDIA L40 + 2×NVIDIA A40，各46068MiB |
| Driver | 595.91.07；nvidia-smi显示驱动CUDA能力13.2 |
| CUDA Toolkit | /usr/local/cuda/bin/nvcc，13.2.51；默认PATH不含nvcc |

初查A40设备1、2空闲，后来其他任务分别占约40GiB。第一次GPU1短测被占用检查主动拒绝，未启动训练。成功短测使用物理GPU0 L40（逻辑cuda:0），启动前已有2421MiB占用，显式记录shared_gpu=true。没有终止任何进程或修改GPU设置。资源状态随时间变化，正式实验需重新检查。

## Python与依赖

- base Python3.11.7，NumPy1.26.4，无PyTorch/PyG/DGL。Conda24.1.2、uv0.11.23可用。
- 只读检查现有Conda环境：domain_eval有torch2.11.0+cu130；llama_factory/qwen_grpo/trl_ppo等有torch2.13.0；这些环境均未修改。各自Python与依赖明细见environment_details.json。
- 新建项目 `.venv`（标准venv，非system-site-packages）。实际使用Python3.11.7、torch2.4.1+cu121、PyG2.3.1、torch-scatter2.1.2+pt24cu121、torch-sparse0.6.18+pt24cu121、NumPy1.26.4、SciPy1.13.1、scikit-learn1.5.2、PyYAML6.0.2、ogb1.3.6、tensorboardX2.6.2.2；全部传递依赖见configs/requirements.lock.txt。
- torch来自官方cu121索引；scatter/sparse来自data.pyg.org的torch-2.4.0+cu121轮子页；其余经PyPI安装。`uv pip check`通过。没有安装DGL，也未在已有环境安装包。
- 驱动CUDA能力、系统Toolkit和torch.version.cuda是三个概念；实际模型GPU前向/反向验证CUDA12.1运行库可用。无需改装系统CUDA13.2或编译自定义CUDA扩展。
- 运行环境不是论文历史环境；复现结论须说明软件版本差异。

## 网络与数据

git clone GitHub成功；arXiv v3 PDF下载成功（1,015,787字节、11页）；download.pytorch.org、data.pyg.org、PyPI依赖安装成功。Cora的8个Planetoid原始文件通过GitHub下载、processed缓存写入成功。说明这些实际端点本次可达，不保证所有外网或以后持续可达。

论文提取工具pdftotext缺失，初始提取调用失败；独立环境安装pypdf5.1.0后已成功提取为docs/paper/GraphMAE.txt。未安装系统工具。

## 实际Smoke Test

成功结果：`results/raw/M0/smoke_20261112T040413960326Z/summary.json`。
控制台：`results/raw/M0/validation_20261112T040413067248Z/smoke_console.txt`。

| 检查/测量 | 实际结果 |
|---|---|
| 数据 | Cora2708节点、1433特征、7类 |
| 边 | 10556条非自环有向边条目，加2708自环后13264；勿直接与论文原始边数5429混用 |
| public split | 140训练、500验证、1000测试；互不交叠 |
| 特征 | 非零行和为1；全部有限 |
| 掩码 | floor(0.5×2708)=1354；无重复；未选节点特征保持原值 |
| 模型 | 官方512维、4heads、2层GAT encoder和1层GAT decoder，未改源码 |
| 前向/反向 | loss有限，梯度有限且非零，mask token被优化更新 |
| embedding | [2708,512]且有限 |
| 训练预算 | seed0，20轮预训练+20轮probe；正式训练标记false |
| 初次独立前向loss | 1.0025923252（诊断抽样后） |
| 预训练loss | 0.9990495443 → 0.5295440555 |
| 线性评估 | 最后一轮test Accuracy=0.730；val最高0.764（从0计数epoch9）对应test Accuracy=0.743 |
| encoder冻结 | probe前后state_dict完全一致 |
| 下载/处理数据耗时 | 10.645秒 |
| 20轮预训练 | 0.22154秒，含GPU同步 |
| 20轮probe | 0.10874秒，含GPU同步 |
| 去前2轮epoch中位耗时 | 0.008015秒 |
| 本进程峰值张量allocated/reserved | 509.07/616.00MiB；不含其他进程与全部CUDA上下文 |

原始逐轮loss和lr在epochs.json；有效配置在effective_config.json；原始与processed数据SHA256在summary.json。上述Accuracy仅是运行诊断，20轮未收敛，不作为论文84.2%的复现结果，更不作为改进结论。

官方PyG路径通过，因此正式选择锁定PyG分支。DGL分支仅审计，无运行验证。

## 报告编译环境

PATH中未找到xelatex、latexmk、bibtex、pdflatex、tectonic；fc-list没有中文字体输出。现已提供ctexart+XeLaTeX的报告源文件与自动附录机制，未生成main.pdf，不能声称编译通过。真实构建状态在reports/build_status.json。

后续方案：在已配TeX Live的个人电脑或课程允许的LaTeX环境中编译完整项目（需要XeLaTeX、BibTeX、ctex/Fandol、fvextra、natbib等）；命令见README。若希望服务器安装用户级TeX，须在后续阶段明确处理，此次不修改系统。M0静态结构检查不代替实际编译与版面复核。

## 算力、时间与存储估算

已测20轮的稳态epoch约8ms；粗估1500轮纯预训练约12秒。将20轮probe线性外推300轮约1.6秒；官方每200轮额外probe共7次、最终1次，约13秒。加上导入、首次CUDA、日志和负载差异，预留每seed1–5分钟、5种子5–25分钟；课程M1预留1小时更稳妥。此为外推预算，不是已经跑过1500轮的测量。

短测使用共享L40，调度/频率/缓存和完整训练可能改变耗时；A40速度未测，不能声称同速。M2若3–4个α组×5种子，预留1–2小时；Citeseer为可选扩展，必须重新测时。

建议单GPU预算2GiB显存（实测张量峰值约0.5GiB，留上下文及波动余量），CPU4线程、内存2–4GiB即可。环境及wheel缓存预留15GiB；多种子checkpoint/图表预留2GiB。收尾时实测venv占5.23GiB，Cora数据15.58MiB，第三方源码及bundle约31.63MiB；详细字节数由results/raw/M0/final_check.json记录（未计共享uv缓存，缓存不随项目源码强制提交）。

## 尚存限制

LaTeX无法实际编译；DGL未运行；GPU并非独占；源码种子函数存在拼写错误且scatter可能不完全确定；PyG旧版加载processed数据有torch.load FutureWarning（不影响本次运行，固定版本避免行为漂移）；正式重复实验和度数策略均未执行。M0到此暂停。

## M1执行补充（2026-11-13，以上保留M0历史）

复用原venv，53包依赖检查通过，没有安装或升级Python/系统依赖。模型与数据版本不变。最初四张卡满载，未启动训练；后续GPU2连续三次0%利用率、约45488MiB空闲后正式运行。五个独立worker均使用物理GPU2 NVIDIA A40（UUID GPU-ea2c34c5-d296-26b5-ecb1-36cc6c7f5f18），CUDA_VISIBLE_DEVICES设为UUID，内部cuda:0，以PID匹配nvidia-smi实际设备；五次启动均非共享。各周期Probe前保存GPU监测快照。

实测完整五种子平均worker耗时26.09秒、合计130.46秒，含预训练、7次周期Probe、最终Probe及导入/加载/保存。每seed峰值allocated472.70MiB、reserved592MiB，不含全部CUDA上下文和其他进程。正式数据在results/raw/M1/cora/20261113T065001179740Z，取代M0对正式实验耗时的外推估计。

绘图复用base Matplotlib3.8.0和NumPy1.26.4，不对训练venv安装绘图包；Noto Sans CJK SC字体和许可证保存在figures/M1/assets，来源/哈希见plot_metadata.json。报告按本阶段用户要求本地编译，不把服务器TeX缺失当作M1阻塞。scatter非完全确定性及5种子统计限制保留。
