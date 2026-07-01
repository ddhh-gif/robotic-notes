**Neural Ordinary Differential Equations 完整推导笔记**

这份笔记综合整理 `neural_ode_slides.pdf`、`Neural_ODE_中文翻译.md` 与论文 `1806.07366v5.pdf`。slides 给出了主线，中文翻译补齐了正文公式，论文 PDF 的附录 A、B、C 则给出了两个最关键证明：瞬时变量替换公式与伴随灵敏度方法。我们要理清的逻辑不是“Neural ODE 是一种新网络层”这么简单，而是三条数学线索如何汇合：第一，ResNet 是 Euler 离散化，连续极限给出 ODE 网络；第二，反向传播可以写成伴随 ODE，从而不必穿过求解器内部；第三，normalizing flow 的 log-determinant 在连续极限下变成 Jacobian trace，从而得到 Continuous Normalizing Flow。

**阅读路线**

我们先把残差网络看成数值积分方法，因为这解释了为什么“层”可以连续化。接着定义 Neural ODE 的前向传播，并说明 ODE solver 作为 black-box primitive 的意义。然后进入最重要的训练问题：如果输出 $z(t_1)$ 是由求解器给出的，损失 $L(z(t_1))$ 如何对初值 $z(t_0)$ 和参数 $\theta$ 求导？这会导向伴随状态 $a(t)$ 和增广反向 ODE。最后，我们把同一套连续时间思想用于密度模型，证明瞬时变量替换公式 $\frac{\partial \log p(z(t))}{\partial t}=-\operatorname{tr}\frac{\partial f}{\partial z(t)}$，并解释 CNF、FFJORD 与 Latent ODE 为什么自然出现。

**变量表**

| 符号                                  |                             类型或形状 | 含义                                                      |
| ----------------------------------- | --------------------------------: | ------------------------------------------------------- |
| $t$                                 |                                标量 | 连续时间变量，也可理解为连续化后的网络深度                                   |
| $t_0$                               |                                标量 | ODE 初始时间，通常对应输入层或初始潜变量时间                                |
| $t_1$ 或 $T$                         |                                标量 | ODE 终止时间，通常对应输出层                                        |
| $h_t$                               |                             $R^D$ | 离散网络第 $t$ 层隐藏状态，论文中离散索引也写作 $t$                          |
| $z(t)$                              |                             $R^D$ | 连续时间隐藏状态或流模型状态                                          |
| $z(t_0)$                            |                             $R^D$ | 初始状态，监督学习中常由输入 $x$ 给出                                   |
| $z(t_1)$                            |                             $R^D$ | 终止状态，作为模型输出或送入后续读出层                                     |
| $D$                                 |                               正整数 | 状态维度或潜变量维度                                              |
| $f$                                 | $R^D\times R\times \Theta\to R^D$ | ODE 右端项或向量场，通常由神经网络参数化                                  |
| $\theta$                            |                              参数向量 | 向量场 $f(z,t,\theta)$ 的可学习参数                              |
| $L$                                 |                              标量函数 | 训练损失，输入可以是终点状态或多个观测状态                                   |
| $a(t)$                              |           $R^{1\times D}$ 或 $R^D$ | 伴随状态，定义为 $a(t)=\frac{\partial L}{\partial z(t)}$        |
| $a_\theta(t)$                       |               $R^{1\times\theta}$ | 损失对时间 $t$ 处参数副本 $\theta(t)$ 的伴随量                        |
| $a_t(t)$                            |                                标量 | 损失对时间变量的伴随量，用于求 $\frac{dL}{dt_0}$ 和 $\frac{dL}{dt_1}$   |
| $\frac{\partial f}{\partial z}$     |                   $R^{D\times D}$ | 向量场关于状态的 Jacobian                                       |
| $\frac{\partial f}{\partial\theta}$ |               $R^{D\times\theta}$ | 向量场关于参数的 Jacobian                                       |
| $\operatorname{ODESolve}$           |                                函数 | 黑箱 ODE 求解器，输入初值、向量场、时间区间和参数                             |
| $\operatorname{NFE}$                |                              非负整数 | Number of Function Evaluations，即求解器评估 $f$ 的次数           |
| $p(z(t))$                           |                              密度函数 | 状态 $z(t)$ 在时间 $t$ 的概率密度                                 |
| $\log p(z(t))$                      |                                标量 | CNF 中沿流演化的 log-density                                  |
| $T_\epsilon$                        |                                映射 | 时间推进 $\epsilon$ 后的短时变换，$z(t+\epsilon)=T_\epsilon(z(t))$ |
| $J$                                 |                                矩阵 | 某个变换或向量场的 Jacobian                                      |
| $\operatorname{tr}(J)$              |                                标量 | 矩阵 $J$ 的迹，即对角线元素之和                                      |
| $\det(J)$                           |                                标量 | 矩阵 $J$ 的行列式                                             |
| $\epsilon$                          |                          小正数或随机向量 | 在极限证明中表示小时间步；在 Hutchinson 估计中表示随机探针                     |
| $q_\phi$                            |                               分布族 | Latent ODE 中的识别网络或变分后验                                  |
| $z_{t_i}$                           |                             $R^D$ | 时间序列在观测时间 $t_i$ 的潜在状态                                   |
| $x_{t_i}$                           |                              数据变量 | 时间 $t_i$ 处的观测值                                          |
| $\lambda(z(t))$                     |                               正标量 | 非齐次泊松过程的事件强度函数                                          |

**核心公式表**

| 位置            | 公式                                                                                                      | 作用                                        |
| ------------- | ------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| ResNet 更新     | $h_{t+1}=h_t+f(h_t,\theta_t)$                                                                           | 离散层的残差递推                                  |
| Euler 方法      | $z(t+\Delta t)=z(t)+\Delta t f(z(t),t,\theta)$                                                          | ODE 的显式一阶离散化                              |
| Neural ODE    | $\frac{dh(t)}{dt}=f(h(t),t,\theta)$                                                                     | 连续深度模型定义                                  |
| ODE 解         | $z(t_1)=z(t_0)+\int_{t_0}^{t_1}f(z(t),t,\theta)dt$                                                      | 前向传播的积分形式                                 |
| 黑箱求解器         | $z(t_1)=\operatorname{ODESolve}(z(t_0),f,t_0,t_1,\theta)$                                               | 工程实现中的模型 primitive                        |
| 损失            | $L(z(t_1))=L(\operatorname{ODESolve}(z(t_0),f,t_0,t_1,\theta))$                                         | 训练目标依赖 ODE 终点                             |
| 伴随定义          | $a(t)=\frac{\partial L}{\partial z(t)}$                                                                 | 连续时间反向传播的状态变量                             |
| 伴随 ODE        | $\frac{da(t)}{dt}=-a(t)^\top\frac{\partial f(z(t),t,\theta)}{\partial z}$                               | 梯度随时间的反向动力学                               |
| 初值梯度          | $\frac{\partial L}{\partial z(t_0)}=a(t_0)$                                                             | 反向积分终点给出输入梯度                              |
| 参数梯度          | $\frac{dL}{d\theta}=-\int_{t_1}^{t_0}a(t)^\top\frac{\partial f(z(t),t,\theta)}{\partial\theta}dt$       | 反向积分中累计参数梯度                               |
| 增广反向系统        | $\frac{d}{dt}[z,a,g]=[f,-a^\top\frac{\partial f}{\partial z},-a^\top\frac{\partial f}{\partial\theta}]$ | 一次求解同时重构状态、传播伴随、累计参数梯度                    |
| 终止时间梯度        | $\frac{dL}{dt_1}=a(t_1)f(z(t_1),t_1,\theta)$                                                            | 完整伴随算法中对终止时间求导                            |
| 离散变量替换        | $\log p(z_1)=\log p(z_0)-\log\left                                                                      | \det\frac{\partial f}{\partial z_0}\right |
| 瞬时变量替换        | $\frac{\partial\log p(z(t))}{\partial t}=-\operatorname{tr}\frac{df}{dz(t)}$                            | CNF 的核心密度方程                               |
| Planar CNF    | $\frac{dz(t)}{dt}=uh(w^\top z(t)+b)$                                                                    | 平面流的连续版本                                  |
| Planar CNF 密度 | $\frac{\partial\log p(z(t))}{\partial t}=-u^\top\frac{\partial h}{\partial z(t)}$                       | 利用外积 trace 等于内积                           |
| 多隐藏单元 CNF     | $\frac{dz(t)}{dt}=\sum_{n=1}^M f_n(z(t))$                                                               | 宽连续流的动力学                                  |
| 多隐藏单元密度       | $\frac{d\log p(z(t))}{dt}=\sum_{n=1}^M\operatorname{tr}\frac{\partial f_n}{\partial z}$                 | trace 线性性带来线性代价                           |
| Hutchinson 估计 | $\operatorname{tr}(J)=E_\epsilon[\epsilon^\top J\epsilon]$                                              | 随机无偏估计 trace                              |
| Latent ODE 先验 | $z_{t_0}\sim p(z_{t_0})$                                                                                | 时间序列潜在初值                                  |
| Latent ODE 轨迹 | $z_{t_1},...,z_{t_N}=\operatorname{ODESolve}(z_{t_0},f,\theta_f,t_0,...,t_N)$                           | 在任意观测时间输出潜状态                              |
| Latent ODE 观测 | $x_{t_i}\sim p(x                                                                                        | z_{t_i},\theta_x)$                        |
| 泊松时间似然        | $\log p(t_1,...,t_N)=\sum_i\log\lambda(z(t_i))-\int_{t_{start}}^{t_{end}}\lambda(z(t))dt$               | 建模观测时间本身                                  |

**第一条主线：为什么 ResNet 像 ODE solver**

我们先问一个看似朴素的问题：如果神经网络越来越深，每一层做的改变越来越小，极限里会发生什么？残差网络的单层更新是 $h_{t+1}=h_t+f(h_t,\theta_t)$。这已经很像 Euler 方法，只是 Euler 方法通常写作 $z(t+\Delta t)=z(t)+\Delta t f(z(t),t,\theta)$。当 $\Delta t=1$ 时，这两个式子几乎一样；当我们显式保留小步长时，残差块可以写成 $h_{k+1}=h_k+\Delta t f(h_k,t_k,\theta)$。

现在令 $t_k=t_0+k\Delta t$。如果层数增加而 $\Delta t$ 变小，那么差商 $\frac{h_{k+1}-h_k}{\Delta t}$ 就逼近导数 $\frac{dz(t)}{dt}$。于是残差网络的连续极限是 $\frac{dz(t)}{dt}=f(z(t),t,\theta)$。这就是 Neural ODE 的第一层含义：它不是简单地“把网络写成微分方程”，而是把残差网络解释为某个连续动力系统的数值离散化。

这个观点也解释了 slides 中那张“Residual Networks interpreted as an ODE Solver”的图。普通 ResNet 是固定步长、固定层数的离散求解器；Neural ODE 则把步长选择交给成熟 ODE solver。求解器可以根据误差容差动态决定在哪些位置评估 $f$，因此模型有效深度不再是人手写死的层数，而是由动力学复杂度和容差共同决定的 $\operatorname{NFE}$。

**Neural ODE 的前向传播**

Neural ODE 不直接给出 $y=F(x)$，而是先令 $z(t_0)=x$，再求解初值问题 $\frac{dz(t)}{dt}=f(z(t),t,\theta)$，最后把 $z(t_1)$ 作为输出或送入分类 head。积分形式为 $z(t_1)=z(t_0)+\int_{t_0}^{t_1}f(z(t),t,\theta)dt$。在程序中，这一切被包装为 $z(t_1)=\operatorname{ODESolve}(z(t_0),f,t_0,t_1,\theta)$。

这一定义带来三个重要变化。第一，模型的计算量是自适应的：简单样本或平滑动力学需要较少函数评估，复杂样本或刚性动力学需要更多函数评估。第二，速度和精度可以通过容差显式交换，而不是只能通过改网络结构交换。第三，训练时不能再天真地把求解器内部每一步都当成普通网络层保存下来，因为自适应求解器的内部轨迹可能很长，隐式求解器还可能包含内层优化，这正是伴随方法要解决的问题。

**第二条主线：为什么需要伴随方法**

现在我们遇到真正的训练问题。损失是 $L(z(t_1))=L(\operatorname{ODESolve}(z(t_0),f,t_0,t_1,\theta))$，我们需要对 $z(t_0)$ 和 $\theta$ 求导。直接反向传播穿过 ODE solver 当然可以想象，但它会要求保存前向求解器的所有中间状态，并且求解器越精细，计算图越长。Neural ODE 的论文选择把求解器视为黑箱，然后使用伴随灵敏度方法。

伴随方法的关键定义是 $a(t)=\frac{\partial L}{\partial z(t)}$。这句话很短，但含义很大：在离散神经网络中，我们有每一层 hidden state 的梯度；在连续深度网络中，我们不再有一层一层的梯度，而是有一条随时间连续变化的梯度轨迹 $a(t)$。如果能写出 $a(t)$ 满足的微分方程，就能像求解前向状态那样求解反向梯度。

**证明一：伴随 ODE**

**Scratch Work.** 我们想证明 $a(t)$ 满足 $\frac{da(t)}{dt}=-a(t)^\top\frac{\partial f}{\partial z}$。这看起来像连续版链式法则，所以应该从离散链式法则出发。对一个很小的时间步 $\epsilon>0$，短时流映射把 $z(t)$ 送到 $z(t+\epsilon)$。如果记这个短时映射为 $T_\epsilon$，那么 $z(t+\epsilon)=T_\epsilon(z(t))$。根据链式法则，$a(t)=a(t+\epsilon)\frac{\partial T_\epsilon}{\partial z(t)}$。因此，只要把 $T_\epsilon$ 的一阶展开代入，再取极限，就应该得到伴随 ODE。

**Proof Idea.** 证明的结构是：先用 ODE 的 Taylor 展开写出短时映射的 Jacobian，再把它放进链式法则，最后用导数定义让 $\epsilon$ 趋于 $0$。

**Proof.** 令 $z(t)$ 是满足 $\frac{dz(t)}{dt}=f(z(t),t,\theta)$ 的状态轨迹，其中 $z(t)$ 属于 $R^D$，参数 $\theta$ 属于参数空间 $\Theta$。定义伴随状态 $a(t)=\frac{\partial L}{\partial z(t)}$。对每个小正数 $\epsilon$，短时 ODE 流给出 $z(t+\epsilon)=T_\epsilon(z(t),t)$。由 ODE 的一阶 Taylor 展开，有 $T_\epsilon(z(t),t)=z(t)+\epsilon f(z(t),t,\theta)+O(\epsilon^2)$。因此，短时映射关于 $z(t)$ 的 Jacobian 为 $\frac{\partial T_\epsilon}{\partial z(t)}=I+\epsilon\frac{\partial f(z(t),t,\theta)}{\partial z(t)}+O(\epsilon^2)$。

根据链式法则，损失对当前状态的梯度可以由稍后状态的梯度拉回，因此 $a(t)=a(t+\epsilon)\frac{\partial T_\epsilon}{\partial z(t)}$。把上一段的 Jacobian 展开代入，得到 $a(t)=a(t+\epsilon)(I+\epsilon\frac{\partial f}{\partial z}+O(\epsilon^2))$。整理这个式子可得 $a(t+\epsilon)-a(t)=-\epsilon a(t+\epsilon)\frac{\partial f}{\partial z}+O(\epsilon^2)$。两边除以 $\epsilon$，并令 $\epsilon\to0^+$，得到 $\frac{da(t)}{dt}=-a(t)\frac{\partial f(z(t),t,\theta)}{\partial z(t)}$。如果把 $a(t)$ 视为列向量，则同一公式会写成相应的转置形式；论文正文中常写作 $\frac{da(t)}{dt}=-a(t)^\top\frac{\partial f(z(t),t,\theta)}{\partial z}$。因此，伴随状态满足所需的连续反向传播方程。

**Moral.** 伴随 ODE 不是一个神秘的新技巧，它就是链式法则在无穷小时间步上的极限。离散网络中，梯度从第 $k+1$ 层传到第 $k$ 层；Neural ODE 中，梯度从时间 $t+\epsilon$ 传回时间 $t$，然后令 $\epsilon$ 趋于零。

**参数梯度从哪里来**

有了 $a(t)$，参数梯度就可以用积分表示。直觉是：在每个时间点 $t$，参数 $\theta$ 通过改变向量场 $f(z(t),t,\theta)$ 来改变后续轨迹；而 $a(t)$ 衡量当前状态改变会对最终损失造成多大影响。所以时间 $t$ 对参数梯度的贡献密度是 $a(t)^\top\frac{\partial f(z(t),t,\theta)}{\partial\theta}$。沿正向时间写，可以表示为 $\frac{dL}{d\theta}=\int_{t_0}^{t_1}a(t)^\top\frac{\partial f}{\partial\theta}dt$。论文为了配合反向积分，把它写成等价形式 $\frac{dL}{d\theta}=-\int_{t_1}^{t_0}a(t)^\top\frac{\partial f}{\partial\theta}dt$。

实现时，我们把累计梯度也作为一个状态变量。令 $g(t)$ 表示累计的参数梯度通道，那么反向求解时使用增广状态 $[z(t),a(t),g(t)]$。对应的动力学为 $\frac{d}{dt}[z,a,g]=[f(z,t,\theta),-a^\top\frac{\partial f}{\partial z},-a^\top\frac{\partial f}{\partial\theta}]$。从 $t_1$ 开始，初始增广状态是 $[z(t_1),\frac{\partial L}{\partial z(t_1)},0]$，反向求解到 $t_0$ 后得到 $a(t_0)=\frac{\partial L}{\partial z(t_0)}$ 和 $g(t_0)=\frac{dL}{d\theta}$。

**完整伴随算法的逻辑**

论文 Algorithm 1 的最简版本可以用一句话描述：把原状态、伴随状态和参数梯度拼接成一个更大的 ODE 系统，然后从终点反向积分回起点。输入包括参数 $\theta$、起始时间 $t_0$、终止时间 $t_1$、最终状态 $z(t_1)$ 和终点梯度 $\frac{\partial L}{\partial z(t_1)}$。算法先设 $s_0=[z(t_1),\frac{\partial L}{\partial z(t_1)},0_{|\theta|}]$，再定义增广动力学 $[f,-a^\top\frac{\partial f}{\partial z},-a^\top\frac{\partial f}{\partial\theta}]$，最后调用 $\operatorname{ODESolve}(s_0,\operatorname{aug\_dynamics},t_1,t_0,\theta)$。

论文 Appendix C 给出了更完整版本，它还计算对 $t_0$ 和 $t_1$ 的梯度。终止时间梯度来自终点状态随终止时间的瞬时变化，因此 $\frac{dL}{dt_1}=\frac{\partial L}{\partial z(t_1)}f(z(t_1),t_1,\theta)$。如果把时间 $t$ 也看成一个状态，且满足 $\frac{dt(t)}{dt}=1$，再把参数 $\theta$ 看成常量状态，且满足 $\frac{d\theta(t)}{dt}=0$，就能把对状态、参数和时间的敏感性统一进一个增广伴随系统。

**多个观测时间的处理**

如果损失只依赖终点，伴随反向积分一次即可。如果损失依赖多个时刻，例如 $L=\sum_i L_i(z(t_i))$，则反向传播必须分段。我们从最后一个观测时刻开始向前积分；每到一个观测时刻 $t_i$，就把该处的直接梯度加到伴随状态中，即 $a(t_i)\leftarrow a(t_i)+\frac{\partial L_i}{\partial z(t_i)}$。这和普通计算图中“一个节点收到多条梯度就相加”完全一样，只是节点现在位于连续时间轴上。

这个机制解释了为什么 Neural ODE 特别适合不规则时间序列。观测时间可以是任意的 $t_1,t_2,...,t_N$，求解器只需在这些时间输出状态，反向传播则在这些时间注入梯度。模型不需要把所有序列强制离散到统一网格，也不需要为空缺时间点伪造观测。

**第三条主线：从普通变量替换到瞬时变量替换**

现在转向密度模型。普通 normalizing flow 使用双射 $z_1=F(z_0)$ 变换随机变量，并用变量替换公式计算密度：$\log p(z_1)=\log p(z_0)-\log\left|\det\frac{\partial F}{\partial z_0}\right|$。这个公式强大但昂贵，因为高维 Jacobian determinant 通常需要立方复杂度。很多离散 flow 架构的设计，本质上都是为了让这个 determinant 便宜。

Continuous Normalizing Flow 的想法是把有限变换 $F$ 替换成连续流 $\frac{dz(t)}{dt}=f(z(t),t)$。只要 $f$ 对 $z$ 一致 Lipschitz 且对 $t$ 连续，初值问题就有唯一解。唯一性意味着两条轨迹不会在同一时间交叉，也意味着从 $t_0$ 到 $t_1$ 的整体映射自动可逆。更妙的是，log-density 的变化由瞬时变量替换公式给出：$\frac{\partial\log p(z(t))}{\partial t}=-\operatorname{tr}\frac{df}{dz(t)}$。我们接下来证明它。

**证明二：瞬时变量替换公式**

**Scratch Work.** 我们想把离散变量替换里的 $\log\det$ 变成连续时间里的 trace。一个自然想法是看一个极短时间步 $\epsilon$。在这么短的时间里，流映射几乎是恒等映射加一个小扰动：$z(t+\epsilon)=z(t)+\epsilon f(z(t),t)+O(\epsilon^2)$。它的 Jacobian 近似为 $I+\epsilon\frac{\partial f}{\partial z}$。而我们知道恒等矩阵附近有近似关系 $\log\det(I+\epsilon A)=\epsilon\operatorname{tr}(A)+O(\epsilon^2)$。所以有限步的 log-determinant 除以 $\epsilon$ 后，在极限中应该变成 trace。

**Proof Idea.** 证明就是把普通变量替换公式应用到短时变换 $T_\epsilon$，然后令 $\epsilon\to0^+$。关键工具是 Jacobi 公式或等价的一阶恒等式 $\log\det(I+\epsilon A)=\epsilon\operatorname{tr}(A)+O(\epsilon^2)$。

**Proof.** 令 $z(t)$ 是一个有限连续随机变量，其密度为 $p(z(t))$。假设 $z(t)$ 由 ODE $\frac{dz(t)}{dt}=f(z(t),t)$ 推动，并且 $f$ 对 $z$ 一致 Lipschitz、对 $t$ 连续。对每个小正数 $\epsilon$，记短时变换为 $z(t+\epsilon)=T_\epsilon(z(t))$。由于 ODE 解存在且唯一，这个短时变换在足够小的时间步内是良好定义的。

对短时变换应用离散变量替换公式，可得 $\log p(z(t+\epsilon))=\log p(z(t))-\log\left|\det\frac{\partial T_\epsilon}{\partial z(t)}\right|$。因此，log-density 的时间导数是 $\frac{\partial\log p(z(t))}{\partial t}=\lim_{\epsilon\to0^+}\frac{\log p(z(t+\epsilon))-\log p(z(t))}{\epsilon}$，也就是 $-\lim_{\epsilon\to0^+}\frac{1}{\epsilon}\log\left|\det\frac{\partial T_\epsilon}{\partial z(t)}\right|$。

由 ODE 的 Taylor 展开，短时变换满足 $T_\epsilon(z(t))=z(t)+\epsilon f(z(t),t)+O(\epsilon^2)$。因此它的 Jacobian 满足 $\frac{\partial T_\epsilon}{\partial z(t)}=I+\epsilon\frac{\partial f(z(t),t)}{\partial z(t)}+O(\epsilon^2)$。令 $A=\frac{\partial f(z(t),t)}{\partial z(t)}$。由矩阵恒等式 $\log\det(I+\epsilon A+O(\epsilon^2))=\epsilon\operatorname{tr}(A)+O(\epsilon^2)$，得到 $-\lim_{\epsilon\to0^+}\frac{1}{\epsilon}\log\left|\det\frac{\partial T_\epsilon}{\partial z(t)}\right|=-\operatorname{tr}\frac{\partial f(z(t),t)}{\partial z(t)}$。所以 $\frac{\partial\log p(z(t))}{\partial t}=-\operatorname{tr}\frac{df}{dz(t)}$，这正是瞬时变量替换公式。

**Moral.** 普通 flow 的难点是高维 determinant；连续化之后，每一个无穷小变换都只是“恒等映射加一个小扰动”，而 determinant 在恒等映射附近的一阶变化就是 trace。CNF 的优雅之处就在这里：它不是魔法地消灭了 Jacobian，而是把 log-determinant 的连续极限变成了 trace。

**CNF 的联合 ODE**

有了瞬时变量替换公式，CNF 可以同时求解状态和密度。联合系统写作 $\frac{d}{dt}[z(t),\log p(z(t))]=[f(z(t),t,\theta),-\operatorname{tr}\frac{\partial f}{\partial z(t)}]$。如果从简单先验出发正向积分，就可以生成样本并跟踪 log-density；如果从数据点出发反向积分回先验空间，就可以计算最大似然所需的 base density 和密度校正项。

这个公式与普通 normalizing flow 的差异很重要。离散 flow 必须让每一层既可逆又有便宜的 determinant；CNF 只需指定一个足够正则的向量场，整体可逆性由 ODE 唯一性保证，密度变化由 trace 给出。代价则转移到 ODE 求解和 trace 计算上，所以 FFJORD 等工作进一步引入随机 trace 估计来降低高维成本。

**Planar CNF 作为例子**

论文给出平面流的连续版本 $\frac{dz(t)}{dt}=uh(w^\top z(t)+b)$，其中 $u$ 和 $w$ 是向量，$b$ 是标量，$h$ 是非线性函数。这个向量场的 Jacobian 可以写成外积形式 $\frac{\partial f}{\partial z}=u\frac{\partial h}{\partial z}$。由于外积的 trace 等于相应内积，得到 $\frac{\partial\log p(z(t))}{\partial t}=-\operatorname{tr}(u\frac{\partial h}{\partial z})=-u^\top\frac{\partial h}{\partial z}$。

这个例子说明了 CNF 为什么可以自然扩展宽度。如果向量场是多个子向量场之和 $\frac{dz(t)}{dt}=\sum_{n=1}^M f_n(z(t))$，那么 Jacobian 也是和，trace 由于线性性满足 $\operatorname{tr}(\sum_n\frac{\partial f_n}{\partial z})=\sum_n\operatorname{tr}\frac{\partial f_n}{\partial z}$。所以 log-density 的微分方程也按隐藏单元线性相加。这正是论文说“使用多个隐藏单元具有线性代价”的原因。

**随机无偏 log-density：Hutchinson 估计**

精确 trace 在高维中仍然可能很贵。Hutchinson 估计给出一个简单而关键的事实：如果随机向量 $\epsilon$ 满足 $E[\epsilon]=0$ 且 $E[\epsilon\epsilon^\top]=I$，则对任意矩阵 $J$，有 $E_\epsilon[\epsilon^\top J\epsilon]=\operatorname{tr}(J)$。证明很直接：$E[\epsilon^\top J\epsilon]=E[\operatorname{tr}(\epsilon^\top J\epsilon)]=E[\operatorname{tr}(J\epsilon\epsilon^\top)]=\operatorname{tr}(J E[\epsilon\epsilon^\top])=\operatorname{tr}(J)$。

因此，CNF 中的 $\operatorname{tr}\frac{\partial f}{\partial z}$ 可以用 $\epsilon^\top\frac{\partial f}{\partial z}\epsilon$ 无偏估计。这个量可以通过自动微分中的向量-Jacobian乘积或 Jacobian-vector product 计算，而不必显式形成完整 Jacobian。slides 中的 “Stochastic Unbiased Log Density” 指的就是这一技术路线，它让 FFJORD 能够把 CNF 推向图像级别的生成建模。

**Latent ODE：不规则时间序列的生成模型**

Latent ODE 把每条时间序列解释为一条潜在连续轨迹。生成过程分三步：先采样初始潜变量 $z_{t_0}\sim p(z_{t_0})$；再用 ODE solver 在任意观测时间输出潜状态，$z_{t_1},z_{t_2},...,z_{t_N}=\operatorname{ODESolve}(z_{t_0},f,\theta_f,t_0,t_1,...,t_N)$；最后生成观测 $x_{t_i}\sim p(x|z_{t_i},\theta_x)$。这里 $f$ 是所有序列共享的全局动力学，而 $z_{t_0}$ 是每条序列自己的初始条件。

训练时通常把它放进 VAE 框架。识别网络 $q_\phi(z_{t_0}|x_{t_1},x_{t_2},...,x_{t_N})$ 根据观测序列推断初始潜变量，然后 ODE 从这个初始点生成整条潜在轨迹。ELBO 可以写成 $E_{q_\phi}[\sum_i\log p(x_{t_i}|z_{t_i})]-\operatorname{KL}(q_\phi(z_{t_0}|x_{t_1},...,x_{t_N})||p(z_{t_0}))$。这个模型的优势是观测时间 $t_i$ 可以任意不规则，不必把时间轴离散成固定网格。

**观测时间本身也能建模**

在医疗、交通、事件流等数据中，观测何时发生本身就携带信息。论文用非齐次泊松过程建模事件时间。设事件强度为 $\lambda(z(t))$，则区间 $[t_{start},t_{end}]$ 内事件时间 $t_1,...,t_N$ 的 log-likelihood 为 $\sum_{i=1}^N\log\lambda(z(t_i))-\int_{t_{start}}^{t_{end}}\lambda(z(t))dt$。第一项鼓励模型在真实事件发生处给高强度，第二项惩罚整个区间内的总强度过大。

这个积分项也可以并入 ODE。定义累计强度 $r(t)=\int_{t_{start}}^t\lambda(z(s))ds$，则 $\frac{dr(t)}{dt}=\lambda(z(t))$。因此，我们可以把 $r(t)$ 和 $z(t)$ 拼成增广状态，一次求解同时得到潜在轨迹和泊松似然所需积分。这是 Neural ODE 风格建模的一贯模式：只要某个量可以写成沿轨迹的积分，就可以把它作为额外状态加入 ODE。

**存在唯一性、可逆性与数值误差**

ODE 的存在唯一性是 Neural ODE 和 CNF 的数学地基。Picard-Lindelof 定理告诉我们，如果 $f$ 对状态 $z$ 一致 Lipschitz 且对时间 $t$ 连续，则初值问题存在唯一解。唯一性意味着同一初值不会产生两条不同轨迹，也意味着两条不同轨迹不能在同一时间相交后再分开。对于 CNF，这带来整体映射的可逆性：从 $t_0$ 到 $t_1$ 正向积分，再从 $t_1$ 到 $t_0$ 反向积分，理论上可以恢复原状态。

不过，理论可逆不等于数值上完全无误差。伴随方法在反向传播时需要重构前向轨迹，如果求解器误差积累或动力学很刚性，重构轨迹可能偏离原始前向轨迹。这就是论文局限性部分提到 checkpointing 的原因：可以在前向传播中保存少数中间点，反向时从这些检查点重新积分，以换取更好的数值稳定性。于是 Neural ODE 的内存优势不是绝对免费的，而是在内存、精度和计算之间提供新的折中。

**NFE 与模型复杂度**

slides 反复出现的 $\operatorname{NFE}$ 是理解 Neural ODE 工程表现的关键。前向传播成本大致由 $\operatorname{NFE}_{forward}$ 次 $f$ 的评估组成；反向传播需要求解增广系统，单次评估还包含向量-Jacobian 乘积，因此成本通常更高。训练过程中如果 $\operatorname{NFE}$ 持续上升，往往说明模型学到的动力学越来越复杂，或者变得更刚性，求解器需要更小步长才能满足同一误差容差。

这也解释了后续 Neural ODE 工作为什么关心动力学正则化。一个表达力很强但高度扭曲、震荡或刚性的向量场，理论上可以拟合数据，却会让求解器极慢。Neural ODE 的好模型不只是预测准确，还应该让连续动力学容易求解。这一点和普通深度网络不同：普通网络的计算图固定，Neural ODE 的计算图复杂度本身是模型学出来的。

**逻辑总图**

| 问题 | 离散深度学习答案 | Neural ODE 答案 |
|---|---|---|
| 如何表示深度变换 | 固定层序列 $h_{t+1}=h_t+f(h_t,\theta_t)$ | 连续向量场 $\frac{dz}{dt}=f(z,t,\theta)$ |
| 如何前向传播 | 逐层计算 | 调用 ODE solver 积分 |
| 如何控制计算量 | 选择层数和宽度 | 选择容差，求解器自适应步长 |
| 如何反向传播 | 保存层激活并反传 | 反向求解伴随 ODE |
| 如何做可逆密度模型 | 设计可逆层并计算 log determinant | 用 ODE 唯一性保证可逆，用 trace 计算 log-density |
| 如何处理不规则时间序列 | 插值、分箱或把时间差输入 RNN | 直接在观测时间求解潜在状态 |

