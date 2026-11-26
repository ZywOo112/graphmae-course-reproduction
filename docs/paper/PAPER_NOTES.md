# GraphMAE 阅读笔记

来源：保存于本目录的 arXiv v3，KDD 2022，Hou等。以下为自己的方法解释；论文结果与本机实验分别标识。

## 背景、问题和设计

图对比学习常依赖图增强、负样本或双编码器；生成式图自监督可以通过重建获得表示，但结构重建可能偏向拓扑，直接特征重建易学到恒等映射，MSE对范数/维数敏感，弱解码器难以学到适合分类的表示。GraphMAE结合特征掩码、GNN解码器、重掩码和SCE，不把链接重建作为主要目标。

设 G=(V,E)，N=|V|，A为邻接矩阵，X∈R^(N×F)。目标是在没有标签的预训练阶段学习 f_E，使下游少量标签训练的分类器受益。Cora采用传导学习：所有节点特征和拓扑在预训练时可见，测试标签不可用。

## Masking（论文§3.2 Q2）

每轮均匀无放回抽取 M⊂V，|M|=floor(rN)。输入为

\[
\tilde x_i=\begin{cases}x_{\mathrm{mask}},& i\in M,\\x_i,&i\notin M.\end{cases}
\]

实际实现将少量masked节点换成随机节点特征，Cora replace_rate=0.05，以缓和预训练/推断差异。token可学习、初值为0；masked节点仍参与全图消息传递，不像某些视觉MAE那样从encoder删除token。论文讨论均匀采样有助于避免邻域全遮蔽或全可见的偏置；这直接构成度数策略需要检验的风险。

## GNN Encoder

\[
H=f_E(A,\tilde X),\qquad
h_i^{(l+1)}=\mathop{\Vert}_{k=1}^{K}\sigma\!\left(\sum_{j\in\mathcal N(i)\cup\{i\}}a_{ij}^{(l,k)}W^{(l,k)}h_j^{(l)}\right).
\]

其中a_ij由邻域内softmax归一化的LeakyReLU注意力分数产生。注意力不是度数采样概率。官方Cora是2层GAT、总隐藏维512（每head128，4heads）、PReLU、特征/注意力dropout。默认无residual、无norm，不把模型名字等同于任意GAT超参数。

## Re-mask 与 Decoder（§3.2 Q3）

代码先线性映射 R=HW，再将同一M中节点的表示设为0：

\[
\tilde r_i=\begin{cases}0,&i\in M,\\r_i,&i\notin M,\end{cases}
\qquad Z=f_D(A,\tilde R).
\]

论文用decoder mask token表述；锁定源码具体用零向量。单层GAT decoder恢复F维特征，使被掩码节点依赖邻域表示。注意未掩码节点的隐表示已在encoder阶段融合周围信息，re-mask不等同于删除所有关于masked节点的间接信息。decoder只在预训练使用；下游推断不用decoder、不掩码。

## Scaled Cosine Error（§3.2 Q4，式2）

\[
\mathcal L_{\mathrm{SCE}}=\frac1{|M|}\sum_{i\in M}\left(1-\frac{x_i^\top z_i}{\|x_i\|_2\|z_i\|_2}\right)^\gamma,\qquad\gamma\ge1.
\]

代码F.normalize沿特征维归一化（含epsilon保护），再求(1-dot)^gamma均值。仅M中节点计损失，不重建所有节点或边。余弦缓和向量范数变化；指数抑制误差小于1的易例相对贡献，不能笼统说所有误差都减小。Cora γ=3，源码参数名alpha_l；与改进策略指数α是两个独立参数。

## 下游评估（§4.1）

固定训练得到的encoder，用完整X、不掩码生成H。线性分类器 softmax(HW_c+b)，训练集交叉熵；验证集选择分类器epoch，测试集报告对应得分。官方训练结束返回最后一轮和验证选择的两个测试Accuracy，主比较应预先指定后者。Cora public split为140训练、500验证、1000测试，剩余节点仅参加无监督图学习。

单标签多分类且所有类别计入时，Micro-F1=Accuracy；论文表1只有PPI使用多标签Micro-F1，其他数据集报告Accuracy。论文用20次初始化，本课程首批5次不能冒称相同重复次数。论文表1 Cora84.2±0.4%、Citeseer73.4±0.4%均属于论文结果。

## Degree-aware Masking 可行性假设（尚未实现）

\[
w_i=(d_i+1)^\alpha,\quad |M|=\lfloor rN\rfloor,\quad
M\sim\operatorname{WeightedWithoutReplacement}(w,|M|).
\]

d_i为去自环、无向去重图的度数。α>0偏向高阶、α=0均匀、α<0偏向低阶；固定总预算便于比较。可使用torch.multinomial(weights,k,replacement=False)，但归一化权重不是无放回抽样最终边际纳入概率，不应写Pr(i∈M)=w_i/sum(w)。α很大需稳定计算避免溢出。

可能收益：高阶节点影响更多消息传递，重建它们或产生更有挑战性的监督。可能损害：hub被同时掩码会削弱大量邻居的信息，训练目标向hub重加权、低阶节点监督减少。该策略会改变未加权masked损失的期望目标；不加重要性校正是本假设的组成部分，不能只声称“采样加速但目标不变”。不能未经检索与实验证明科研新颖性；课程自主实验足以。

实验计划：统一r=0.5，固定其余参数；5配对种子；α∈{0,0.5,1}，必要时加-1诊断；按原图度数分位分组测试Accuracy、实际每节点掩码频率、分组重建误差和可见邻居比例。所有阈值从图结构确定，不用测试标签调参。观察平均差值和种子方差，即使下降也完整报告。

## 报告需要的补充材料

M1补全实际loss/时间/多种子结果与论文差异；M2补全α、度数组与公平性分析；M3核对GCN/GAT/自监督相关工作的一手引用，补齐模型图和公式符号表。M0尚未产生正式复现或改进结论。
