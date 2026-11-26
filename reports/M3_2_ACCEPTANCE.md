# M3.2 验收：正文、公式、参考文献与图表

日期：2026-11-16。状态：**M3.2已完成，暂停；未进入M3.3。** 未训练/推断模型，未改变实验数据、配置、统计或原图，未安装TeX或执行PDF编译。

## 正文与源文件

主文件为`reports/main.tex`，沿用原ctexart、A4单栏、2.5cm页边距、正常正文小四字号及1.25倍行距。以九个`reports/sections/*.tex`组织正文，便于后续局部修改。摘要与关键词已写，目录不另占一页；数学、图片、表格按章编号，正文使用label/ref/eqref/cite，不硬编码图表号。

| 章节文件 | 实际内容 |
|---|---|
| 01_introduction.tex | 图表示与自监督动机、原论文贡献和本项目复现/采样假设的区分 |
| 02_related.tex | GCN/GAT、DGI、VGAE、BERT/视觉MAE与GraphMAE的联系；只保留主题相关研究 |
| 03_method.tex | 图与特征定义、均匀掩码、token/随机替换、GAT注意力、多头聚合、投影/Re-mask/解码、SCE、冻结线性Probe；张量维度表 |
| 04_protocol.tex | 固定Commit、硬件软件、Cora三种边计数、划分与transductive标签隔离、固定配置、7次周期Probe、验证选模、种子与样本SD |
| 05_reproduction.tex | M1全部五seed结果及时间；Loss、论文参考比较、82.2%低分保留；5次与20次初始化及环境差异 |
| 06_degree_masking.tex | 唯一邻居度数、权重、固定1354预算、四组20次、独立CPU采样与shadow RNG控制、M2控制组重新训练 |
| 07_results.tex | 四组验证与测试前选参、五对最终测试、分组真实掩码率和分类、Loss/时间/显存 |
| 08_discussion.tex | 机制解释均作为假设；单图/少种子、M1/M2随机协议、最后一次A40推断、GPU非完全确定性等限制 |
| 09_conclusion.tex | 复现结论、自主假设的负结果及适用边界，不增加新性能主张 |

完整公式含16个数学标签（15个equation/align环境，含一个双行align），重要符号与维度均在邻近文字定义。原论文的一般描述与Cora锁定实现分开说明，包括：decoder为1层；Re-mask为置零；1286 token/67替换/1个取整保留；SCE只在M上计算且gamma=3；degree_alpha不是alpha_l；归一化采样权重不等于最终无放回边际概率。

当前正文源及TikZ输入含约8390个汉字（静态正则计数，含图注/表格等，不等同正文净字数）。采用紧凑论文结构而非项目实施日志。目标约10—15页，但服务器未排版，**不能承诺实际页数或声称已经达到页数目标**；必要时本地调整图宽/浮动体和重复呈现，不能删除必要种子或限制说明。

## 数值与图片

- M1：83.98%均值、1.056882个百分点样本SD；论文参考84.2%±0.4%、20次初始化；本地仅5次。
- M2四组验证均值80.04%、79.60%、78.52%、76.16%；正向候选中先按验证选择0.25，且全部正向候选均不如控制。
- M2最终控制84.20%±0.37个百分点，0.25为82.92%±1.67个百分点，平均配对差-1.28个百分点；五对全部列出，0.5/1.0测试明确写“未测试”。
- 分组人数、正确数和实际掩码率来自既有CSV/预测，不虚构高度数受益。末100轮SCE为0.402125与0.414893，不写成Loss改善。

正文实际使用9张图、8张表。7张实验PNG直接复用现行M1两张和M2五张；官方架构图直接使用原图的同字节JPEG副本`reports/figures/graphmae_official.jpg`，图注引用GraphMAE，不重绘原作者框架；自主方法仅新画一张中文TikZ流程`reports/figures/degree_masking_scheme.tex`。全部来源及表格/公式依据见`reports/M3_2_SOURCES.md`，图片路径相对项目，不依赖服务器绝对目录。所有图表在正文都有引用与解释。

## 参考文献

`reports/references.bib`共7条且全部实际引用：GraphMAE、GCN、GAT、DGI、VGAE、BERT、视觉MAE。新增后六条的标题、作者、年份/发表形式已通过作者arXiv、ACL Anthology或CVF官方来源核对；不使用未核实标题或虚构作者。ICLR年份与预印本年份区分；VGAE按预印本并注明workshop，未冒称主会；视觉MAE按正式CVPR 2022而非早期2021预印本引用。采用natbib数字引用和unsrtnat，按首次引用排序。

已核实的DOI保留；ICLR会议DOI及视觉MAE的正式DOI未从成功取得的一手元数据确认，留空而不猜填。CVF原始HTML直开403但其官方页搜索结果提供完整BibTeX；OpenReview返回浏览器校验，ICLR归属由作者arXiv comments及本地GraphMAE参考表交叉核对。GraphMAE ACM DOI由本地原论文首页核实，不能声称其本次跳转页面访问成功。完整核验链接、日期、访问限制均见来源清单。

CCF官网总目录搜索索引已明确列SIGKDD于A类，直接页面仍405；本阶段没有伪造目录快照。可本地进一步人工核对原目录及实际课程截止日期的五年窗口。正文的学术结论没有依赖错误访问记录。

## 静态检查结果

执行命令（项目根目录）：

```bash
.venv/bin/python reports/check_sources.py
```

输出：`reports/M3_2_STATIC_CHECK.json`，状态`passed_static_only`，错误0。核对内容包括：

- 9个正文章节、9张图、8张表、42个标签，无重复标签、缺失ref或未引用图表；7个BibTeX键均存在且被引用。
- 所有input和图片存在；花括号及begin/end环境栈配对；无正文TODO、pending或服务器绝对路径。
- 官方框架JPEG与原文件SHA256相同，格式及3658×728尺寸按文件头核实；7张实验图为真实PNG。
- 856项受保护实验/图片/脚本/配置等清单记录匹配；两官方Git工作树清洁、Commit正确。经授权更新的正文/BibTeX/管理文件不拿旧报告哈希错误地判断为实验篡改。

这不是TeX编译器或完整语法解释器。没有运行xelatex、latexmk、bibtex、模型、附录生成器或对话导出器，也没有生成main.pdf。现有`reports/build_status.json`是M0历史状态，未追改成M3检查结果。

## 保留给M3.3与人工复核

完整附录入口仍在main.tex，`\includeFullAppendixfalse`默认关闭，避免把旧M0附录当成本阶段已完成附件；原`appendices/generated.tex`及manifest均未改动。M3.3再统一导入全部自编源文件（须新增本次sections、TikZ和check_sources.py等）、真实主要Prompt/最终回答、Skill/工作流、第三方与电子附件清单，修正旧导出脚本的commentary和M0副作用。误发的其他项目讨论不进入GraphMAE阶段任务附件。

用户本地应在完整项目的`reports/`目录使用XeLaTeX/BibTeX或latexmk编译，复核：作者姓名学号；图内字号及官方英文框架缩放；表格宽度；数学长行；TikZ中文；参考文献重音/长URL；浮动体顺序、页数及打印可读性。源文件保留Fandol和常见宏包，不需服务器特定字体或TeX路径。参考命令为`latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex`；本阶段没有执行。

M3.2到此暂停，不自动整理完整附录或进入M3.3。
