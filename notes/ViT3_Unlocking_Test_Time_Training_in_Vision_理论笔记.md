**ViT3: Unlocking Test-Time Training in Vision 理论笔记**

本文讨论的核心问题是：如何把视觉 Transformer 中代价为 $O(N^2)$ 的 Softmax attention 改写为一种线性复杂度 $O(N)$ 的序列建模机制，同时保留足够强的表达能力。论文的关键观点是，attention 可以被理解为一种“根据当前输入上下文临时构造出来的模型”，而 Test-Time Training，简称 TTT，则把这个临时模型显式写成一个可训练的内层模型 $F_W$，并在测试时用当前样本的 key-value 对对它进行少量自监督更新，再用更新后的参数处理 query。

**变量表**

| 符号 | 形状或类型 | 含义 |
|---|---:|---|
| $x$ | $R^{N \times C}$ | 输入 token 序列，$N$ 是 token 数，$C$ 是通道维度或 embedding 维度 |
| $N$ | 标量 | 序列长度；在视觉任务中通常对应 patch/token 数 |
| $C$ | 标量 | 输入 token 的通道维度 |
| $d$ | 标量 | 单个 attention head 或 TTT inner module 的特征维度 |
| $W_Q$ | $R^{C \times d}$ | query 投影矩阵 |
| $W_K$ | $R^{C \times d}$ | key 投影矩阵 |
| $W_V$ | $R^{C \times d}$ | value 投影矩阵 |
| $Q$ | $R^{N \times d}$ | query 矩阵，$Q=xW_Q$ |
| $K$ | $R^{N \times d}$ | key 矩阵，$K=xW_K$ |
| $V$ | $R^{N \times d}$ | value 矩阵，$V=xW_V$ |
| $Q_i$ | $R^{1 \times d}$ | 第 $i$ 个 query token |
| $K_j$ | $R^{1 \times d}$ | 第 $j$ 个 key token |
| $V_j$ | $R^{1 \times d}$ | 第 $j$ 个 value token |
| $O$ | $R^{N \times d}$ | 输出 token 矩阵 |
| $O_i$ | $R^{1 \times d}$ | 第 $i$ 个输出 token |
| $\sigma(\cdot)$ | 函数 | 按行计算的 Softmax 函数 |
| $\phi(\cdot)$ | 函数 | 线性 attention 中的 kernel feature map |
| $F_W$ | $R^d \to R^d$ | TTT 的内层模型，参数为 $W$ |
| $W$ | 参数集合 | 内层模型 $F_W$ 的参数，也可以看成上下文压缩后的“快速权重” |
| $W_0$ | 参数集合 | 内层模型的初始参数，由外层网络学习得到 |
| $W^*$ | 参数集合 | 经过 test-time inner update 后的内层模型参数 |
| $D$ | 集合 | key-value 构成的内层训练集，$D=\{(K_i,V_i)\}_{i=1}^N$ |
| $B$ | 标量 | 内层训练的 mini-batch 大小；论文发现视觉中 $B=N$ 即 full-batch 效果较好 |
| $K_B$ | $R^{B \times d}$ | 一个 inner mini-batch 中的 key |
| $V_B$ | $R^{B \times d}$ | 一个 inner mini-batch 中的 target value |
| $\hat V_B$ | $R^{B \times d}$ | 内层模型对 $K_B$ 的预测，$\hat V_B=F_W(K_B)$ |
| $L(\hat V_B,V_B)$ | 标量 | 内层自监督损失，使 $F_W(K_i)$ 预测对应的 $V_i$ |
| $\eta$ | 标量 | 内层学习率 |
| $G$ | 参数梯度 | 内层更新梯度，$G=\partial L(\hat V_B,V_B)/\partial W$ |
| $i,j$ | 索引 | $i$ 常表示 token 索引，$j$ 可表示 token 或通道索引，依上下文而定 |
| $R^{a \times b}$ | 形状记号 | 表示 $a$ 行 $b$ 列实数矩阵 |

**核心公式表**

| 模块 | 公式 | 复杂度或作用 |
|---|---|---|
| Softmax attention 单 token 形式 | $O_i=\sum_{j=1}^N \frac{\exp(Q_iK_j^\top)}{\sum_{j=1}^N \exp(Q_iK_j^\top)}V_j$ | 对每个 query 与所有 key 计算相似度，总体复杂度 $O(N^2d)$ |
| Softmax attention 矩阵形式 | $O=\sigma(QK^\top)V$ | $QK^\top$ 产生 $N \times N$ attention map |
| Softmax attention 的 MLP 视角 | $O=\sigma(QW_1)W_2$，其中 $W_1=K^\top$，$W_2=V$ | 可看作宽度为 $N$ 的两层上下文 MLP |
| 线性 attention | $O_i=\frac{Q_i(\sum_{j=1}^N K_j^\top V_j)}{Q_i(\sum_{j=1}^N K_j^\top)}$ | 通过结合律避免显式 $N \times N$ 矩阵 |
| 忽略归一化的线性 attention | $O=Q(K^\top V)$ | 把 $K,V$ 压缩为 $d \times d$ 线性权重 |
| TTT 内层预测 | $\hat V_B=F_W(K_B)$ | 用 key 预测 value，形成自监督训练目标 |
| TTT 内层更新 | $W \leftarrow W-\eta \frac{\partial L(\hat V_B,V_B)}{\partial W}$ | 在测试时根据当前样本更新内层模型 |
| TTT 输出 | $O=F_{W^*}(Q)$ | 使用更新后的内层模型处理 query |
| Dot product loss | $L=-\frac{1}{B\sqrt d}\sum_{i=1}^B \hat V_iV_i^\top$ | 论文实验中有效的内层损失之一 |
| MSE loss | $L=\frac{1}{2B\sqrt d}\sum_{i=1}^B(\hat V_i-V_i)(\hat V_i-V_i)^\top$ | 混合二阶导不为零，适合 TTT |
| MAE loss | $L=\frac{1}{B\sqrt d}\sum_{i=1}^B\lVert \hat V_i-V_i\rVert_1$ | 混合二阶导几乎处处为零，效果差 |

**从 Softmax attention 到上下文 MLP**

传统 Softmax attention 的输入是 $Q,K,V \in R^{N \times d}$，其输出为 $O=\sigma(QK^\top)V$。由于 $QK^\top \in R^{N \times N}$，每个 query 都要和所有 key 交互，因此主要计算量来自 $N \times N$ 的相似度矩阵，复杂度随序列长度呈二次增长。对于高分辨率视觉任务，$N$ 会随着图像 token 数迅速变大，这正是 ViT 在长序列视觉建模中的瓶颈。

论文的一个重要解释是，可以把 $K^\top$ 和 $V$ 看成临时构造出的两层 MLP 参数。因为 $O=\sigma(QK^\top)V$ 可以改写为 $O=\sigma(QW_1)W_2$，其中 $W_1=K^\top$，$W_2=V$。从这个角度看，Softmax attention 并不是固定参数的 MLP，而是一个由当前上下文 $K,V$ 动态生成的 MLP；其隐藏层宽度为 $N$，激活函数为 row-wise Softmax。这个解释揭示了 Softmax attention 强大的原因：它为每个输入上下文构造了一个依赖当前样本的动态模型；也揭示了它昂贵的原因：这个动态模型的隐藏宽度直接等于序列长度 $N$。

**线性 attention 的压缩视角**

线性 attention 的思路是把 Softmax 的非线性相似度替换成可分解的 kernel 形式，使计算可以从 $(QK^\top)V$ 重排为 $Q(K^\top V)$。如果暂时忽略归一化项，线性 attention 就是 $O=Q(K^\top V)$。这里 $K^\top V \in R^{d \times d}$ 可以被看作一个由上下文生成的线性层权重 $W=K^\top V$，所以输出就是 $O=QW$。

这个形式的优势是显然的：先计算 $K^\top V$ 的复杂度约为 $O(Nd^2)$，再计算 $Q(K^\top V)$ 的复杂度也是 $O(Nd^2)$，在 $d$ 固定或远小于 $N$ 时是线性序列复杂度。问题也同样显然：所有 $N$ 个 key-value token 被压缩进一个 $d \times d$ 的线性矩阵中，而且压缩方式只是一次矩阵乘法 $K^\top V$。这使线性 attention 的表达能力受限，尤其在视觉任务中可能丢失复杂的空间关系和语义结构。

**TTT 的核心重构**

TTT 可以看作对线性 attention 的推广：线性 attention 把 $K,V$ 压缩进一个线性层 $W=K^\top V$，而 TTT 把 $K,V$ 压缩进一个更一般的内层模型 $F_W:R^d\to R^d$ 的参数 $W$ 中。具体地，当前序列产生一个小型训练集 $D=\{(K_i,V_i)\}_{i=1}^N$，内层模型需要学习映射 $K_i \mapsto V_i$。在一个 mini-batch 上，先计算 $\hat V_B=F_W(K_B)$，再通过 $W \leftarrow W-\eta \partial L(\hat V_B,V_B)/\partial W$ 更新内层参数，最终得到 $W^*$，并输出 $O=F_{W^*}(Q)$。

这里的关键变化是，attention 不再被限制为 Softmax 生成的 $N$ 宽 MLP，也不再被限制为线性 attention 的 $d \times d$ 线性层，而是可以选择任意线性复杂度的内层模型，例如 MLP、门控线性单元或卷积模块。只要 $F_W$ 的前向、反向和 query 推理都对 $N$ 线性，整个 TTT layer 就可以保持 $O(N)$ 的序列复杂度，同时获得比普通线性 attention 更强的非线性表达能力。

**为什么内层损失的混合二阶导数重要**

论文最重要的理论分析之一是：内层损失 $L$ 不能只看预测误差本身，还要看它是否能把外层梯度传给 value projection $W_V$。因为 $V=xW_V$，而 TTT 的内层更新依赖 $V$，外层训练时需要反向传播穿过内层更新步骤。令内层更新梯度为 $G=\partial L(\hat V_B,V_B)/\partial W$，则 $W_V$ 接收到的梯度与 $\partial G/\partial W_V$ 有关。

根据链式法则，论文给出关系 $\frac{\partial G}{\partial W_V}=\frac{\partial^2L(\hat V_B,V_B)}{\partial W\partial W_V}=\frac{\partial \hat V_B}{\partial W}\cdot\frac{\partial^2L(\hat V_B,V_B)}{\partial \hat V_B\partial V_B}\cdot\frac{\partial V_B}{\partial W_V}$。其中 $\partial \hat V_B/\partial W$ 来自内层模型，$\partial V_B/\partial W_V$ 来自 value projection，真正由损失函数决定的是中间项 $\partial^2L/\partial \hat V_B\partial V_B$。如果这个混合二阶导数为零或接近零，那么外层训练就很难通过内层更新影响 $W_V$，导致 TTT 学不好。

**内层损失推导表**

| 损失 | 定义 | 混合二阶导数 | 结论 |
|---|---|---|---|
| Dot product | $L=-\frac{1}{B\sqrt d}\sum_i \hat V_iV_i^\top$ | $\frac{\partial^2L}{\partial V_{ij}\partial \hat V_{ij}}=-\frac{1}{B\sqrt d}$ | 不为零，能传递外层梯度 |
| MSE | $L=\frac{1}{2B\sqrt d}\sum_i(\hat V_i-V_i)(\hat V_i-V_i)^\top$ | $\frac{\partial^2L}{\partial V_{ij}\partial \hat V_{ij}}=-\frac{1}{B\sqrt d}$ | 不为零，适合 TTT |
| RMSE | $L=\sqrt{\frac{1}{B\sqrt d}\sum_i(\hat V_i-V_i)(\hat V_i-V_i)^\top}$ | $-\frac{1}{B\sqrt d\sqrt S}+\frac{(\hat V_{ij}-V_{ij})^2}{B^2dS^{3/2}}$，其中 $S=\frac{1}{B\sqrt d}\sum_i(\hat V_i-V_i)(\hat V_i-V_i)^\top$ | 一般不为零，但数值形式更复杂 |
| MAE | $L=\frac{1}{B\sqrt d}\sum_i\lVert \hat V_i-V_i\rVert_1$ | 当 $\hat V_{ij}\ne V_{ij}$ 时为 $0$ | 几乎处处不能提供所需二阶信号 |
| Smooth L1 | $L=\frac{1}{B\sqrt d}\sum_{i,j}\ell(\hat V_{ij}-V_{ij})$，$\ell(x)=\frac{1}{2}x^2$ 若 $\lvert x\rvert<1$，否则 $\lvert x\rvert-\frac{1}{2}$ | 小误差区域为 $-\frac{1}{B\sqrt d}$，大误差区域为 $0$ | 部分区域信号消失，表现不如 Dot/MSE |

**Dot product loss 的推导**

Dot product loss 定义为 $L=-\frac{1}{B\sqrt d}\sum_{i=1}^B \hat V_iV_i^\top$。对元素 $\hat V_{ij}$ 求偏导，得到 $\frac{\partial L}{\partial \hat V_{ij}}=-\frac{1}{B\sqrt d}V_{ij}$。再对 $V_{ij}$ 求导，得到 $\frac{\partial^2L}{\partial V_{ij}\partial \hat V_{ij}}=-\frac{1}{B\sqrt d}$。这个值是非零常数，因此外层优化可以稳定地通过内层更新影响 value projection。

**MSE loss 的推导**

MSE loss 定义为 $L=\frac{1}{2B\sqrt d}\sum_i(\hat V_i-V_i)(\hat V_i-V_i)^\top$。对 $\hat V_{ij}$ 求导，得到 $\frac{\partial L}{\partial \hat V_{ij}}=\frac{1}{B\sqrt d}(\hat V_{ij}-V_{ij})$。再对 $V_{ij}$ 求导，得到 $\frac{\partial^2L}{\partial V_{ij}\partial \hat V_{ij}}=-\frac{1}{B\sqrt d}$。因此 MSE 和 Dot product loss 一样，具有非零混合二阶导数，这解释了它们在实验中表现较好。

**MAE loss 为什么不适合 TTT**

MAE loss 定义为 $L=\frac{1}{B\sqrt d}\sum_i\lVert \hat V_i-V_i\rVert_1$。对 $\hat V_{ij}$ 求导得到 $\frac{\partial L}{\partial \hat V_{ij}}=\frac{1}{B\sqrt d}\operatorname{sign}(\hat V_{ij}-V_{ij})$。当 $\hat V_{ij}\ne V_{ij}$ 时，符号函数对 $V_{ij}$ 的导数为 $0$，所以 $\frac{\partial^2L}{\partial V_{ij}\partial \hat V_{ij}}=0$。这意味着 MAE 虽然可以作为普通预测损失，但在 TTT 这种需要反向传播穿过内层更新的结构中，会切断关键的外层学习信号。

**为什么视觉中 full-batch inner training 更合适**

论文发现，对于视觉 TTT，单个 epoch 的 full-batch 内层训练，即 $B=N$，通常效果最好。原因是视觉图像 token 之间没有自然语言那样强烈的因果顺序。如果把视觉 token 切成多个 mini-batch 依次更新，早期 batch 会先改变内层模型，影响后续 batch 的梯度；后续 batch 又可能覆盖早期 batch 的信息。这种顺序依赖对语言建模可能有帮助，因为语言本身是因果序列，但对视觉任务可能引入不自然的扫描偏置。

因此，视觉中的合理默认设置是使用所有 key-value token 一次性构成内层训练集，执行一次 full-batch 梯度下降。这个设置有两个好处：第一，它以非因果方式同时利用所有视觉 token，符合图像的二维全局结构；第二，它计算更简单、更稳定，也更适合并行化。

**内层学习率的作用**

TTT 的内层更新为 $W \leftarrow W-\eta \partial L/\partial W$，所以 $\eta$ 控制当前样本对内层模型参数的改写幅度。论文实验发现，在视觉任务中 $\eta=1.0$ 是有效的默认值。若 $\eta$ 太小，$W$ 几乎没有根据当前 key-value 对发生适配，TTT 退化为接近固定模型；若 $\eta$ 太大，内层模型可能被过度更新，导致外层训练不稳定。

一个有用的特殊情况是线性内层模型 $F_W(x)=xW$ 加 MSE loss。此时内层梯度更新项可以写成 $\eta\frac{\partial L}{\partial W}=\eta K^\top(KW-V)$，也可以等价地写成 $\tilde K^\top(\tilde K W-\tilde V)$，其中 $\tilde K=\sqrt\eta K$，$\tilde V=\sqrt\eta V$。这说明在理想化线性情形下，调整学习率类似于缩放 $K,V$。但在真实网络中，归一化、初始化、非线性和外层优化都会破坏这种简单等价，因此 $\eta$ 仍然是重要超参数。

**内层模型容量与深度的区别**

论文区分了“增大内层模型宽度”和“加深内层模型深度”。增大宽度，例如把两层 MLP 的 hidden dimension 从 $d$ 提高到 $4d$，通常能提升性能，因为 TTT 的内层模型可以用更高容量压缩当前上下文，比 $d \times d$ 线性层表达力更强。

但加深内层模型并不一定有效。论文发现，从单层 FC 到两层 MLP、三层 MLP，测试性能并不随深度单调提高，反而可能下降。原因不在于深层模型理论表达力不足，而在于它更难优化。外层训练要学习一个好的初始参数 $W_0$，使它经过少数 inner update 后变成适合当前样本的 $W^*$；内层训练又要在极少步数内稳定压缩 $K,V$。当 $F_W$ 太深时，外层初始化学习和内层梯度传播都会变难，出现梯度消失、梯度爆炸或 underfitting。

**卷积作为视觉 TTT 内层模型**

视觉任务具有强局部结构，因此论文认为卷积很适合作为 TTT 的内层模型。若 $F_W$ 是卷积，TTT 实际上是在用当前样本的全局 key-value 上下文更新卷积核参数，再用更新后的卷积核处理 query。这样，卷积核的权重包含全局上下文信息，而卷积操作本身保留局部感受野归纳偏置。

对于普通点对点的 TTT 训练集，样本是 $D=\{(K_i,V_i)\}_{i=1}^N$。当内层模型是 $3\times3$ 卷积时，可以理解为训练样本变成 $D=\{(K_i^{loc},V_i)\}_{i=1}^N$，其中 $K_i^{loc}$ 是以第 $i$ 个 token 为中心的局部 key 邻域。于是 TTT 不仅学习单 token 的 key-value 映射，也学习局部空间邻域到中心 value 的映射，这更符合视觉建模。

**ViT3 block 的设计总结**

基于上述实验和分析，论文提出 ViT3 的 TTT block。内层训练采用单 epoch、full-batch、学习率 $1.0$，损失使用 dot product loss。内层模型由两个主要模块组成：一个是简化门控线性单元 $F_1(x)=FC(x)\odot SiLU(FC(x))$，另一个是深度可分离卷积 $F_2(x)=DWConv(x)$。前者比普通线性层表达力更强且相对容易优化，后者注入视觉局部结构。

在多头结构中，ViT3 通常让一个 head 使用卷积型内层模型 $F_2$，其余 head 使用门控线性模块 $F_1$。这种做法可以看成对全局 token 压缩和局部视觉归纳偏置的折中：大部分 head 负责高效的全局上下文建模，一个 head 显式负责局部结构整合。最终 ViT3 可以作为标准 attention block 的替代模块，插入非层级 ViT、层级 H-ViT3 或 DiT 图像生成模型中。

**复杂度理解**

Softmax attention 的瓶颈是 $QK^\top \in R^{N\times N}$，所以随 $N$ 二次增长。线性 attention 通过 $K^\top V$ 把上下文压缩为 $d\times d$ 状态，序列复杂度为 $O(N)$，但表达力有限。TTT 的复杂度取决于内层模型 $F_W$：一次内层训练通常包含对 key 的前向 $\hat V=F_W(K)$、一次反向传播计算 $\partial L/\partial W$，以及对 query 的前向 $O=F_{W^*}(Q)$。如果反向传播约等于两次前向计算，那么一个 TTT layer 大约相当于同结构外层模块的四次前向计算，但仍然对 $N$ 保持线性。

因此，TTT 的实际 trade-off 是：它比普通线性 attention 更贵一些，因为需要内层更新；但它比 Softmax attention 在长序列上更可扩展，因为不构造 $N\times N$ attention map；同时它比固定线性状态更有表达力，因为上下文被压缩进一个可学习、可非线性、可卷积的内层模型中。

**一句话总结**

ViT3 的理论核心可以概括为：Softmax attention 是由 $K,V$ 临时生成的宽度为 $N$ 的上下文 MLP，线性 attention 是把 $K,V$ 直接压缩成 $d\times d$ 线性权重，而 TTT 则把 $K,V$ 作为测试时自监督训练集，用少量梯度步把它们压缩进可设计的内层模型 $F_W$，从而在 $O(N)$ 复杂度下获得更强的视觉序列建模能力。
