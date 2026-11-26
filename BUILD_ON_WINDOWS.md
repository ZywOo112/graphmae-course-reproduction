# Windows本地编译与交付核查

本工程已准备正文和完整附录源码，**服务器未运行TeX编译，当前不提供或声称已有最终main.pdf**。只编译报告无需Python、CUDA、Git或重新运行实验。

## 1. 安装与解压

使用支持XeLaTeX的TeX Live，建议完整安装以减少缺包。安装后重新打开PowerShell，确认`xelatex --version`、`bibtex --version`及`latexmk --version`可用；未识别命令时先检查TeX Live的Windows可执行目录是否已加入PATH。不要在报告目录里复制同名可执行文件替代正确安装。

完整解压交付ZIP。ZIP最外层只有`GraphMAE/`，应得到：

```text
GraphMAE/
  BUILD_ON_WINDOWS.md
  README.md
  SKILL.md
  AGENT_WORKFLOW.md
  FILE_MANIFEST.json
  SHA256SUMS.txt
  scripts/  configs/  data/  results/  figures/  third_party/  agent_logs/
  reports/
    main.tex
    references.bib
    sections/
    appendices/
    figures/
```

例如放到`D:\Coursework\GraphMAE\`。保留所有目录层级，不能只复制main.tex或reports文件夹；附录从reports相对访问`../scripts/`等文件。工程文件为UTF-8，若编辑器询问编码，选择UTF-8并避免将中文文件另存为系统ANSI编码。路径含空格时在命令中使用双引号。

## 2. 编译命令

在PowerShell中进入reports：

```powershell
Set-Location "D:\Coursework\GraphMAE\reports"
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
```

latexmk会根据引用需要调用BibTeX并重复XeLaTeX。若没有latexmk，可使用：

```powershell
xelatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

成功后本地生成`reports/main.pdf`。使用XeLaTeX，不要误用pdfLaTeX。已启用完整附录，不需要再运行Python生成器或对话导出器；本地也不需要访问原Codex会话绝对路径。保留TeX/BibTeX、章节、TikZ与附录源文件，最终PDF是本地编译验收后追加的交付物。

## 3. 常见问题

- **缺少宏包**：通过TeX Live Manager（tlmgr）安装错误日志中点名的包。所用主要包为ctex、Fandol、amsmath/amssymb、booktabs、graphicx、setspace、tabularx、TikZ/PGF、caption、placeins、natbib、fvextra、hyperref。fvextra依赖fancyvrb。安装包后重新编译，勿删去附录来规避报错。
- **中文字体缺失或方框**：主文件已设`fontset=fandol`，优先使用TeX Live自带Fandol，不依赖服务器字体。检查`ctex`和`fandol`是否安装完整，并确保用XeLaTeX。若自行替换字体，应使用本机合法安装的字体，并另查正文与等宽代码中的中文是否都能显示。
- **引用显示问号或编号未更新**：检查BibTeX是否执行，阅读main.blg；重跑BibTeX和两遍XeLaTeX。新增引用时确认references.bib有对应键。`Label(s) may have changed`通常意味着还需一遍编译。
- **找不到图片或附录文件**：确认完整解压，并从reports目录编译；检查是否错误地多加/删掉GraphMAE一级目录。官方框架副本是`reports/figures/graphmae_official.jpg`；原仓库的fig.png实际为JPEG，已用正确扩展名副本引用，不必转换图片。
- **长URL或表格越界**：搜索main.log中的`Overfull \\hbox`，定位页码后检查。优先调整列宽、局部tabcolsep或合理换行，不用删掉数据来缩短表格。第7章度数组表已局部使用footnotesize和4pt列间距，仍需实际PDF复核。
- **页数较多**：完整代码与真实任务原文会使附录明显长于正文，这是课程完整性要求。可以微调附录行距/标题间距，不能用省略号、选段或截断代码来减少页数。

报告代码字号7.5pt/9pt行距、对话8.5pt/10pt，附录普通文字9pt/11pt。fvextra设置自动换行，无文件级强制新页、无浮动体或minipage包住长代码，允许自然跨页。若发现长行溢出，应检查`breaklines`、`breakanywhere`是否仍启用以及等宽中文字体；不要把代码缩放成无法阅读的整页图片。

## 4. 字体与实验复现边界

交付包不包含.otf/.ttf等字体二进制文件。正文和代码使用TeX Live字体，实验PNG已渲染中文，不需要安装绘图字体即可插入PDF。若确需重新绘图，请按`figures/M1/plot_metadata.json`中的Noto Sans CJK SC官方来源及许可获取字体，保存到该元数据指定的assets位置；原始LICENSE与校验记录已保留。可从[官方Noto CJK仓库](https://github.com/notofonts/noto-cjk)获取，遵守其LICENSE。不将字体包缺失误判为实验结果缺失，也不为编译报告重跑训练。

训练环境不随包提供，依赖锁、安装脚本和原始配置均在。Linux/CUDA实验不能据Windows编译成功就声称可原样一键重跑；必要时按照README重建独立、版本匹配的实验环境，另行获得实验授权。历史JSON、日志和真实Prompt中保留的服务器绝对路径属于原始证据，不是本地TeX依赖，不能为了移植而改写历史记录。

两个官方源码工作树全部文件与MIT LICENSE均提供；机器专属`.git`指针和对象库不打包，`third_party/GraphMAE.bundle`保留完整历史。若需要Git元数据，可在另一个新目录执行`git clone path/to/GraphMAE.bundle GraphMAE-history`，再检出README所列Commit；编译与阅读源码不需要此步骤。

## 5. 文件校验

下载后可先核对ZIP的SHA256与同名`.zip.sha256`：

```powershell
Get-FileHash "D:\Downloads\GraphMAE_M3_3_实际时间戳.zip" -Algorithm SHA256
```

完整解压后，ZIP内`FILE_MANIFEST.json`列出所有项目载荷文件的相对路径、字节数与SHA256；`SHA256SUMS.txt`还覆盖FILE_MANIFEST本身。两份索引不互相递归列出自身，整个ZIP的外部SHA256覆盖全部内容。可在GraphMAE根目录核对载荷：

```powershell
$manifest = Get-Content -Raw -Encoding UTF8 ".\FILE_MANIFEST.json" | ConvertFrom-Json
$bad = @()
foreach ($entry in $manifest.files.PSObject.Properties) {
    $p = Join-Path (Get-Location) $entry.Name.Replace('/', '\')
    if (-not (Test-Path -LiteralPath $p)) { $bad += $entry.Name; continue }
    $actual = (Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $entry.Value.sha256) { $bad += $entry.Name }
}
$bad
```

无输出表示清单中的载荷均匹配。修改作者信息或本地编译生成文件后哈希变化属于本地修改，应保留原ZIP作为提交来源。若确需更新电子包，应重新冻结清单，不手工改清单中的数字来掩盖不一致。

## 6. 提交与打印前检查

1. 填写main.tex中的姓名与学号占位符；不改实验数字。
2. 确认九章正文、7条参考文献、附录A—D都出现，主要对话为M0、M1、M2、M3.1、M3.2五对原文。
3. 检查官方横向框架图的文字、图例、坐标轴及误差条；清晰度与实际打印字号都要可读。保留原图，必要时只调整显示宽度。
4. 检查第7章表格未超出页边距，公式编号不重叠，图表与正文引用一致。
5. 抽查附录最长代码行、中文注释、下划线路径、跨页行号；对照电子源码确认末行没有被截断。
6. 按A4、100%实际大小打印若干正文与附录样页，确认7.5pt代码仍可辨认；避免打印驱动再次“缩小以适合页面”。页码、边距、连续分页与图片灰度区分也需检查。
7. 保存本地编译成功的main.pdf与完整源码一起提交。本阶段静态检查通过，不等于以上本地编译和打印验收已经完成。
