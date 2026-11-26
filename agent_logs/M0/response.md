M0 的环境配置、源码审计、GPU 短测和项目初始化已完成，**已暂停，未进入 M1**。尚未完成的是 LaTeX 实际编译：服务器缺少相关工具，未生成 `main.pdf`。

项目入口：[README.md](/home/fuqiang/homework/机器学习课程大作业/GraphMAE/README.md)。

| 项目 | 结果 |
|---|---|
| 服务器 | Ubuntu 24.04.4；104 逻辑 CPU；约 1 TiB 内存；初查磁盘剩余约 422 GiB |
| GPU | 2×L40、2×A40，每张 46068 MiB；本次短测共享物理 GPU 0 |
| 实现选择 | 官方 PyG 分支，Commit `9127f25e01aa5ba1cca68b746284aeb7ab2b4850`；DGL 主分支已审计并完整保留 |
| 独立环境 | 项目 `.venv`；Python 3.11.7、PyTorch 2.4.1+cu121、PyG 2.3.1；53 个包依赖检查通过 |
| Cora | 2708 节点、1433 特征、7 类；140/500/1000 固定划分检查通过 |
| Smoke Test | 前向、反向、参数更新、20 轮预训练及 20 轮线性评估通过；评估期间编码器保持冻结 |
| 诊断结果 | loss：0.9990 → 0.5295；验证集选出的分类器测试 Accuracy：74.3%；峰值张量显存约 509 MiB |

**74.3% 仅是短测诊断，不能视为正式复现结果。** 原始记录见 [summary.json](/home/fuqiang/homework/机器学习课程大作业/GraphMAE/results/raw/M0/smoke_20261112T040413960326Z/summary.json)。

管理文件、论文笔记、Skill、工作流和报告骨架均已建立：

- [源码审计与选择理由](/home/fuqiang/homework/机器学习课程大作业/GraphMAE/docs/SOURCE_AUDIT.md)
- [环境、依赖及资源审计](/home/fuqiang/homework/机器学习课程大作业/GraphMAE/docs/ENVIRONMENT_AUDIT.md)
- [阶段计划与验收标准](/home/fuqiang/homework/机器学习课程大作业/GraphMAE/PROJECT_PLAN.md)
- [当前进展及风险](/home/fuqiang/homework/机器学习课程大作业/GraphMAE/PROGRESS.md)
- [LaTeX 报告骨架](/home/fuqiang/homework/机器学习课程大作业/GraphMAE/reports/main.tex)
- [附录材料清单](/home/fuqiang/homework/机器学习课程大作业/GraphMAE/docs/APPENDIX_MANIFEST.md)

附录已自动导入 13 份完整自编源码、配置及报告源文件，并纳入 Skill、工作流和真实对话材料。第三方源码及可离线恢复的 Git bundle 已保存，官方源码未修改。

真实 M0 Prompt 已保存。归档器已启动，会在**本回答实际写入会话后**保存完整 `response.md` 并刷新附录，避免预写内容冒充真实回答。

主要问题均已记录：官方 `--use_cfg` 会覆盖训练轮数；原空闲 GPU 后被占用；基础环境缺少 PyTorch；报告编译工具缺失。前几项已处理，XeLaTeX、latexmk、BibTeX 及中文字体仍需后续解决；未修改系统环境。

M1 建议先完成报告编译验证，再运行 Cora 的 **5 种子、1500 轮预训练＋300 轮线性评估**。按短测外推，建议预留 1 小时 GPU 窗口；单 GPU 预算约 2 GiB 显存即可。当前独立环境约占 5.23 GiB，建议环境与缓存预留 15 GiB。Degree-aware Masking 尚未实现，其收益仍需 M2 公平对照验证。