**Neural Ordinary Differential Equations Long Form Math 笔记**

这份笔记整理自 `neural_ode_slides.pdf`，主题是 Neural ODE，即用神经网络参数化常微分方程的向量场，并把黑箱 ODE 求解器作为深度学习模型中的可微分模块。核心思想可以用一句话概括：传统深度网络把隐藏状态写成有限层离散递推 $h_{k+1}=h_k+f(h_k,\theta_k)$，Neural ODE 则把层索引连续化，令隐藏状态满足 $\frac{dz(t)}{dt}=f(z(t),t,\theta)$，模型输出由初值问题的解 $z(T)$ 给出。这样做的数学收益包括：模型深度变成连续时间，求解器可以自适应选择步长，反向传播可以通过伴随灵敏度方法以近似 $O(1)$ 内存完成，连续归一化流中的 log-density 变化也可以从昂贵的 log-determinant 变成 trace。

**变量表**

| 符号 | 类型或形状 | 含义 |
|---|---:|---|
| $t$ | 标量 | 连续时间变量，也可理解为连续化后的网络深度 |
| $t_0$ | 标量 | ODE 初始时间，通常对应输入层 |
| $t_1$ 或 $T$ | 标量 | ODE 终止时间，通常对应输出层 |
| $h_k$ | $R^D$ | 离散残差网络第 $k$ 层隐藏状态 |
| $z(t)$ | $R^D$ | 连续时间隐藏状态，Neural ODE 的核心状态变量 |
| $z(t_0)$ | $R^D$ | 初始状态，通常由输入 $x$ 给出 |
| $z(t_1)$ | $R^D$ | 终止状态，通常作为模型输出或后续 head 的输入 |
| $x$ | 数据空间变量 | 输入样本，监督学习中可作为 $z(t_0)$ |
| $y$ | 数据空间变量 | 模型预测输出，有时记作 $y=z(T)$ 或由 $z(T)$ 经读出层得到 |
| $D$ | 标量 | 隐藏状态维度 |
| $f$ | $R^D\times R\times \Theta \to R^D$ | ODE 右端项或向量场，通常由神经网络参数化 |
| $\theta$ | 参数向量 | 向量场 $f(z,t,\theta)$ 的可学习参数 |
| $L$ | 标量函数 | 训练损失，例如分类损失、负对数似然或重构损失 |
| $a(t)$ | $R^{1\times D}$ 或 $R^D$ | 伴随状态，定义为 $a(t)=\frac{\partial L}{\partial z(t)}$ |
| $\frac{\partial f}{\partial z}$ | $R^{D\times D}$ | 向量场关于状态的 Jacobian |
| $\frac{\partial f}{\partial \theta}$ | $R^{D\times |\theta|}$ | 向量场关于参数的 Jacobian |
| $\operatorname{tr}(\cdot)$ | 标量函数 | 矩阵迹，即对角线元素之和 |
| $p(z(t))$ | 密度函数 | 连续归一化流中状态 $z(t)$ 的概率密度 |
| $\log p(z(t))$ | 标量 | 状态沿连续流演化时的 log-density |
| $\epsilon$ | 随机向量 | Hutchinson trace estimator 中的随机探针向量 |
| $\operatorname{NFE}$ | 整数 | Number of Function Evaluations，ODE 求解器调用 $f$ 的次数 |
| $\Delta t$ | 标量 | 离散数值方法中的时间步长 |
| $K$ | 整数 | 离散网络层数或离散 flow 层数，依上下文而定 |
| $M$ | 整数 | 连续归一化流中隐藏单元或子向量场数量 |
| $q_\phi$ | 分布族 | Latent ODE 中的变分后验或识别网络 |
| $z_{t_i}$ | $R^D$ | 时间序列模型在观测时间 $t_i$ 的潜在状态 |
| $\lambda(z(t))$ | 正标量 | 非齐次泊松过程的事件强度函数 |

**核心公式表**

| 主题 | 公式 | 含义 |
|---|---|---|
| ResNet 离散更新 | $h_{k+1}=h_k+f(h_k,\theta_k)$ | 残差块把状态沿某个方向推进一步 |
| Euler 方法 | $z(t+\Delta t)=z(t)+\Delta t f(z(t),t,\theta)$ | ODE 的一阶显式离散化 |
| ResNet 作为 Euler 特例 | $h_{k+1}=h_k+f(h_k,\theta_k)$ 对应 $\Delta t=1$ 的 Euler | 残差网络可看作固定步长 ODE solver |
| Neural ODE 动力学 | $\frac{dz(t)}{dt}=f(z(t),t,\theta)$ | 用神经网络表示连续深度向量场 |
| ODE 解算子 | $z(t_1)=z(t_0)+\int_{t_0}^{t_1}f(z(t),t,\theta)dt$ | 输出由积分轨迹给出 |
| 黑箱求解器形式 | $z(t_1)=\operatorname{ODESolve}(z(t_0),f,t_0,t_1,\theta)$ | 把求解器作为模型 primitive |
| 伴随状态 | $a(t)=\frac{\partial L}{\partial z(t)}$ | 损失对任意时刻状态的敏感性 |
| 伴随 ODE | $\frac{da(t)}{dt}=-a(t)^\top\frac{\partial f(z(t),t,\theta)}{\partial z}$ | 反向时间传播梯度的连续版链式法则 |
| 参数梯度 | $\frac{dL}{d\theta}=-\int_{t_1}^{t_0}a(t)^\top\frac{\partial f(z(t),t,\theta)}{\partial\theta}dt$ | 通过反向积分累计参数梯度 |
| 增广反向系统 | $\frac{d}{dt}[z,a,g]=[f,-a^\top\frac{\partial f}{\partial z},-a^\top\frac{\partial f}{\partial\theta}]$ | 一次反向 ODE 求解同时恢复状态、伴随和参数梯度 |
| 离散变量替换 | $\log p(z_1)=\log p(z_0)-\log|\det\frac{\partial z_1}{\partial z_0}|$ | 普通 normalizing flow 的密度变化 |
| 瞬时变量替换 | $\frac{\partial \log p(z(t))}{\partial t}=-\operatorname{tr}\frac{\partial f}{\partial z(t)}$ | Continuous Normalizing Flow 的核心公式 |
| CNF 联合 ODE | $\frac{d}{dt}[z,\log p(z)]=[f(z,t,\theta),-\operatorname{tr}\frac{\partial f}{\partial z}]$ | 同时求状态轨迹和 log-density |
| Hutchinson 迹估计 | $\operatorname{tr}(J)=E_\epsilon[\epsilon^\top J\epsilon]$ | 用随机估计降低 trace 计算成本 |
| Latent ODE 生成过程 | $z_{t_0}\sim p(z_{t_0})$，$z_{t_1},...,z_{t_N}=\operatorname{ODESolve}(z_{t_0},f,t_0,...,t_N)$，$x_{t_i}\sim p(x|z_{t_i})$ | 连续时间潜变量时间序列模型 |
| 泊松观测时间似然 | $\log p(t_1,...,t_N)=\sum_i\log\lambda(z(t_i))-\int_{t_{start}}^{t_{end}}\lambda(z(t))dt$ | 建模不规则观测时间本身的信息 |

**从残差网络到 ODE 的推导**

残差网络的基本层可以写成 $h_{k+1}=h_k+f(h_k,\theta_k)$。如果把层索引 $k$ 看成时间，把每一层看成沿向量场前进一步，那么这就是显式 Euler 方法 $z(t+\Delta t)=z(t)+\Delta t f(z(t),t,\theta)$ 在 $\Delta t=1$ 时的形式。更一般地，如果网络有很多层，每一层只对状态做很小改变，可以写成 $h_{k+1}=h_k+\Delta t f(h_k,t_k,\theta)$，其中 $t_k=t_0+k\Delta t$。当 $\Delta t\to0$ 且层数趋于无穷时，差商 $\frac{h_{k+1}-h_k}{\Delta t}$ 收敛到导数 $\frac{dz(t)}{dt}$，于是得到连续深度模型 $\frac{dz(t)}{dt}=f(z(t),t,\theta)$。

这说明 Neural ODE 并不是凭空提出的新结构，而是残差网络在层数连续化极限下的自然形式。离散网络显式指定每一层的变换，Neural ODE 则指定一个连续向量场，让 ODE solver 自动选择在哪些时间点评估 $f$，从 $z(t_0)$ 积分到 $z(t_1)$。因此，传统网络的“层数”在 Neural ODE 中不再是固定超参数，而是由求解器的函数评估次数 $\operatorname{NFE}$ 隐式决定。

**Neural ODE 的模型定义**

普通神经网络通常直接定义 $y=F(x,\theta)$。Neural ODE 改为先设定初值 $z(t_0)=x$，再求解初值问题 $\frac{dz(t)}{dt}=f(z(t),t,\theta)$，最后令 $y=z(t_1)$ 或 $y=g(z(t_1))$。其积分形式是 $z(t_1)=z(t_0)+\int_{t_0}^{t_1}f(z(t),t,\theta)dt$。在实现上，模型写成 $z(t_1)=\operatorname{ODESolve}(z(t_0),f,t_0,t_1,\theta)$。

这个定义有三个直接后果。第一，计算路径是自适应的：如果某个输入对应的动力学很平滑，求解器可以用较少函数评估；如果动力学复杂，求解器会自动减小步长并增加 $\operatorname{NFE}$。第二，精度和速度可以显式交换：调松误差容差会更快但更粗糙，调紧容差会更慢但更准确。第三，反向传播不必穿过求解器内部的每个数值步骤，而可以借助伴随方程把 ODE solver 当成黑箱。

**伴随灵敏度方法的目标**

设损失为 $L(z(t_1))$，其中 $z(t_1)$ 是 ODE 解。训练时需要计算 $\frac{\partial L}{\partial z(t_0)}$ 和 $\frac{\partial L}{\partial\theta}$。最直接的办法是记录前向 ODE solver 的每一步，然后像普通网络一样反向传播。但这会带来两个问题：一是自适应求解器可能有很多内部步骤，存储所有中间状态的内存很高；二是隐式求解器内部可能包含优化迭代，直接反向传播会非常复杂。

伴随灵敏度方法的做法是定义 $a(t)=\frac{\partial L}{\partial z(t)}$，即损失对任意时刻状态的梯度。然后推导 $a(t)$ 自身满足的 ODE。只要能从 $t_1$ 反向积分到 $t_0$，就能得到 $a(t_0)=\frac{\partial L}{\partial z(t_0)}$。同时，在反向积分过程中累计参数梯度，就可以得到 $\frac{\partial L}{\partial\theta}$。这个方法本质上是连续时间版本的 reverse-mode autodiff。

**伴随方程的推导**

为了推导伴随 ODE，考虑从时间 $t$ 推进一个无穷小步长 $\epsilon$。由 ODE 定义，有 $z(t+\epsilon)=z(t)+\epsilon f(z(t),t,\theta)+O(\epsilon^2)$。根据链式法则，$a(t)=\frac{\partial L}{\partial z(t)}=\frac{\partial L}{\partial z(t+\epsilon)}\frac{\partial z(t+\epsilon)}{\partial z(t)}$。而 $\frac{\partial z(t+\epsilon)}{\partial z(t)}=I+\epsilon\frac{\partial f}{\partial z}+O(\epsilon^2)$，所以 $a(t)=a(t+\epsilon)(I+\epsilon\frac{\partial f}{\partial z})+O(\epsilon^2)$。

把上式整理为 $a(t+\epsilon)-a(t)=-\epsilon a(t+\epsilon)\frac{\partial f}{\partial z}+O(\epsilon^2)$，两边除以 $\epsilon$ 并令 $\epsilon\to0$，得到 $\frac{da(t)}{dt}=-a(t)^\top\frac{\partial f(z(t),t,\theta)}{\partial z}$。这里符号的正负容易混淆：公式给的是伴随状态随正向时间 $t$ 的导数；实际计算梯度时，我们从 $t_1$ 向 $t_0$ 反向积分这个方程，所以它正好承担反向传播的作用。

**参数梯度的推导**

参数 $\theta$ 对损失的影响来自向量场 $f(z,t,\theta)$。考虑在时间 $t$ 对参数做一个无穷小扰动，状态的无穷小变化会通过 $\frac{\partial f}{\partial\theta}$ 进入后续轨迹。由连续时间链式法则，参数梯度的密度为 $a(t)^\top\frac{\partial f(z(t),t,\theta)}{\partial\theta}$。若沿正向时间从 $t_0$ 积分到 $t_1$，可写成 $\frac{dL}{d\theta}=\int_{t_0}^{t_1}a(t)^\top\frac{\partial f}{\partial\theta}dt$。论文和 slides 常以反向积分形式写作 $\frac{dL}{d\theta}=-\int_{t_1}^{t_0}a(t)^\top\frac{\partial f}{\partial\theta}dt$，二者完全等价，因为改变积分上下限会引入负号。

在实现中，可以把累计参数梯度记为 $g(t)$。若反向从 $t_1$ 到 $t_0$ 积分，增广状态可以写为 $[z(t),a(t),g(t)]$，动力学写成 $\frac{d}{dt}[z,a,g]=[f(z,t,\theta),-a^\top\frac{\partial f}{\partial z},-a^\top\frac{\partial f}{\partial\theta}]$，初始增广状态为 $[z(t_1),\frac{\partial L}{\partial z(t_1)},0]$。求解到 $t_0$ 后，得到 $z(t_0)$、$a(t_0)$ 和累计的 $\frac{\partial L}{\partial\theta}$。

**为什么伴随方法是 $O(1)$ 内存**

普通反向传播需要存储每一层激活，因为计算第 $k$ 层梯度时需要第 $k$ 层的输入。Neural ODE 的伴随方法不存储完整前向轨迹，而是在反向传播时从终点 $z(t_1)$ 反向求解原 ODE，重构需要的 $z(t)$，并同时积分伴随状态 $a(t)$。因此，理论上只需要保存终点状态、时间点、参数和少量求解器状态，内存不随求解器内部步数线性增长。

这里的代价是计算量：反向传播时需要再次求解一个增广 ODE，而且为了获得 $z(t)$ 还要在反向过程中重构轨迹。不过 slides 中强调，经验上反向 pass 的开销并不一定比前向大很多，甚至在一些实验中大约是前向的一半。更重要的是，反向传播不要求知道 ODE solver 内部细节，因此可以支持黑箱求解器，包括显式、隐式和自适应步长方法。

**多观测时间的伴随更新**

如果损失不仅依赖终点 $z(t_1)$，还依赖若干中间观测时间 $z(t_i)$，例如 $L=\sum_i L_i(z(t_i))$，反向积分需要分段进行。具体地，从最后时间点开始向前积分伴随 ODE；每当到达一个观测时刻 $t_i$，伴随状态需要加上该处损失的直接梯度，即 $a(t_i)\leftarrow a(t_i)+\frac{\partial L_i}{\partial z(t_i)}$。这与离散 RNN 或计算图中的反向传播完全一致：如果某个节点直接连到损失，就要把这条路径产生的梯度加到该节点的总 adjoint 上。

这种机制是 Latent ODE 时间序列模型的基础。因为观测可以发生在任意不规则时间 $t_i$，ODE solver 可以直接在这些时间输出潜在状态，而不需要把时间轴强行离散成固定间隔。反向传播时，只需在观测时刻注入对应的局部梯度，再继续反向积分。

**瞬时变量替换公式的背景**

普通归一化流使用双射 $z_1=F(z_0)$ 把简单分布变成复杂分布。密度变化由变量替换公式给出：$\log p(z_1)=\log p(z_0)-\log|\det\frac{\partial F}{\partial z_0}|$。这个公式的问题是，Jacobian determinant 在高维中很贵，通常需要 $O(D^3)$。因此传统 normalizing flow 必须设计特殊结构，例如 coupling layer、自回归结构或低秩变换，使 determinant 可计算。

Continuous Normalizing Flow，简称 CNF，把有限变换 $F$ 改成 ODE 流 $\frac{dz(t)}{dt}=f(z(t),t,\theta)$。如果向量场满足 Lipschitz 条件，则初值问题存在唯一解，并且从 $z(t_0)$ 到 $z(t_1)$ 的映射自动可逆，逆变换只需把 ODE 从 $t_1$ 积分回 $t_0$。关键定理是，连续时间下 log-density 的变化不再需要 log determinant，而只需要 Jacobian trace：$\frac{\partial \log p(z(t))}{\partial t}=-\operatorname{tr}\frac{\partial f}{\partial z(t)}$。

**瞬时变量替换公式的证明**

证明可以从一个小时间步 $\epsilon$ 开始。由 ODE，短时间映射为 $z(t+\epsilon)=z(t)+\epsilon f(z(t),t)+O(\epsilon^2)$。这个映射的 Jacobian 是 $\frac{\partial z(t+\epsilon)}{\partial z(t)}=I+\epsilon\frac{\partial f}{\partial z}+O(\epsilon^2)$。普通变量替换公式给出 $\log p(z(t+\epsilon))=\log p(z(t))-\log|\det(I+\epsilon\frac{\partial f}{\partial z}+O(\epsilon^2))|$。

接下来使用矩阵恒等式 $\log\det(I+\epsilon A)=\epsilon\operatorname{tr}(A)+O(\epsilon^2)$。因此 $\log p(z(t+\epsilon))-\log p(z(t))=-\epsilon\operatorname{tr}\frac{\partial f}{\partial z}+O(\epsilon^2)$。两边除以 $\epsilon$ 并令 $\epsilon\to0$，得到 $\frac{\partial \log p(z(t))}{\partial t}=-\operatorname{tr}\frac{\partial f}{\partial z(t)}$。这个证明展示了为什么连续极限会把 log determinant 简化成 trace：因为每个无穷小变换都接近恒等映射，而恒等映射附近 determinant 的一阶变化正是 trace。

**CNF 的联合求解形式**

有了瞬时变量替换公式，CNF 可以把状态和 log-density 拼成一个联合 ODE：$\frac{d}{dt}[z(t),\log p(z(t))]=[f(z(t),t,\theta),-\operatorname{tr}\frac{\partial f}{\partial z(t)}]$。从简单先验 $p(z(t_0))$ 出发，正向积分可以产生样本并跟踪其密度；从数据 $x=z(t_1)$ 出发，反向积分回 $z(t_0)$ 可以计算最大似然训练所需的 base density 和 log-density correction。

与离散 flow 相比，CNF 的主要优点是架构约束少。离散 flow 必须保证每一层可逆且 log determinant 便宜；CNF 只要向量场满足足够的正则性，整体流映射就由 ODE 唯一性保证可逆，而 log-density correction 只需要 trace。主要代价是需要 ODE 求解器多次评估 $f$，且 trace 的精确计算在高维中仍可能昂贵。

**随机无偏 log-density 与 Hutchinson 估计**

为了降低 trace 计算成本，FFJORD 使用 Hutchinson trace estimator。对任意矩阵 $J$，若随机向量 $\epsilon$ 满足 $E[\epsilon]=0$ 且 $E[\epsilon\epsilon^\top]=I$，则 $E_\epsilon[\epsilon^\top J\epsilon]=\operatorname{tr}(J)$。证明很短：$E[\epsilon^\top J\epsilon]=E[\operatorname{tr}(\epsilon^\top J\epsilon)]=E[\operatorname{tr}(J\epsilon\epsilon^\top)]=\operatorname{tr}(J E[\epsilon\epsilon^\top])=\operatorname{tr}(J)$。

因此，CNF 中的 log-density ODE 可以把 $\operatorname{tr}\frac{\partial f}{\partial z}$ 替换成随机估计 $\epsilon^\top\frac{\partial f}{\partial z}\epsilon$。这个估计是无偏的，并且可以通过自动微分中的向量-Jacobian乘积或 Jacobian-vector product 高效计算，不需要显式构造完整 $D\times D$ Jacobian。slides 中的 “Stochastic Unbiased Log Density” 正是指这一点：log-density 的瞬时变化可以用随机 trace estimator 估计，从而把高维 CNF 扩展到图像生成等任务。

**ODE 求解器作为建模 primitive**

把 ODE solver 当成模型 primitive 的含义是，网络不再显式写出固定数量的层，而是写出一个向量场和一个误差容差。求解器根据局部误差估计决定步长，如果误差过大就缩小步长，如果误差足够小就放大步长。这与传统网络中的固定计算量不同：Neural ODE 的计算量随样本和动力学复杂度变化。

这种自适应计算也带来一个诊断指标，即 $\operatorname{NFE}$。如果训练过程中 $\operatorname{NFE}$ 不断上升，说明模型学到的动力学变得越来越复杂或越来越刚性，求解器需要更多函数评估才能达到相同误差容差。slides 中强调，计算时间主要由评估动力学 $f$ 的次数主导，因此正则化动力学、减少刚性、减少无意义振荡，都是提升 Neural ODE 效率的重要方向。

**ODE 与 RNN 的区别**

RNN 以离散时间递推状态，典型形式是 $h_{k+1}=RNNCell(h_k,x_k,\theta)$。如果观测时间不规则，RNN 通常需要把时间差 $\Delta t$ 作为额外输入，或者把数据插值到固定网格上。Neural ODE 则天然定义在连续时间上，只需要在任意观测时刻调用求解器输出 $z(t_i)$。因此，Latent ODE 特别适合不规则采样时间序列，如医疗记录、事件数据或物理轨迹。

从梯度角度看，普通 RNN 可以学习非常陡峭或刚性的离散动力学，导致梯度爆炸或消失。ODE 模型在满足 Lipschitz 条件时具有连续光滑轨迹，状态随时间不会发生离散跳变。这并不意味着 Neural ODE 不会有优化问题；刚性动力学仍会让求解器变慢，反向重构仍可能有数值误差。但它提供了一种更自然的连续时间归纳偏置。

**Latent ODE 的生成模型推导**

Latent ODE 用一条连续潜在轨迹解释时间序列。生成过程为：先从先验采样初始潜变量 $z_{t_0}\sim p(z_{t_0})$；然后通过 ODE 得到所有观测时间的潜在状态 $z_{t_1},...,z_{t_N}=\operatorname{ODESolve}(z_{t_0},f,\theta_f,t_0,...,t_N)$；最后对每个时间点生成观测 $x_{t_i}\sim p(x|z_{t_i},\theta_x)$。这里 $f$ 是共享的全局动力学，$z_{t_0}$ 是每条序列自己的局部初始条件。

训练时通常使用 VAE 框架。识别网络 $q_\phi(z_{t_0}|x_{t_1},...,x_{t_N})$ 根据观测序列推断初始潜变量，然后用 ODE 生成潜在轨迹并解码观测。对应的 evidence lower bound 可以写成 $E_{q_\phi}[\sum_i\log p(x_{t_i}|z_{t_i})]-\operatorname{KL}(q_\phi(z_{t_0}|x_{t_1},...,x_{t_N})||p(z_{t_0}))$。因为 $z_{t_i}$ 是由 ODE solver 在真实观测时刻输出的，所以模型不需要固定时间网格。

**泊松过程观测时间似然**

在不规则时间序列中，观测何时发生本身可能携带信息。例如病人更严重时更频繁接受检查。Latent ODE 可以用强度函数 $\lambda(z(t))$ 建模事件发生率。对于区间 $[t_{start},t_{end}]$ 内观测时间 $t_1,...,t_N$，非齐次泊松过程的 log-likelihood 是 $\log p(t_1,...,t_N|z(t))=\sum_{i=1}^N\log\lambda(z(t_i))-\int_{t_{start}}^{t_{end}}\lambda(z(t))dt$。

这个公式有直观解释：第一项奖励模型在真实事件发生时间给出高强度，第二项惩罚模型在整个区间内给出过高总强度。由于 $z(t)$ 本来就由 ODE 给出，积分项也可以作为额外状态加入 ODE 一起求解。例如定义 $r(t)=\int_{t_{start}}^t\lambda(z(s))ds$，则 $\frac{dr(t)}{dt}=\lambda(z(t))$。这样一次 ODE solve 就可以同时得到潜在轨迹和事件时间似然所需的积分。

**存在唯一性与可逆性**

Neural ODE 和 CNF 的许多性质依赖 ODE 解的存在唯一性。经典 Picard-Lindelof 定理说明，如果 $f(z,t)$ 对 $z$ 一致 Lipschitz 且对 $t$ 连续，那么初值问题存在唯一解。唯一性意味着两条轨迹不能在相同时间相交，因为如果相交，则交点之后同一初值会对应两条解，违反唯一性。

这个性质直接带来 CNF 的可逆性。给定终点 $z(t_1)$，只要从 $t_1$ 反向积分同一个 ODE 到 $t_0$，就能恢复唯一的起点 $z(t_0)$。因此 CNF 不需要像离散 normalizing flow 那样手工设计可逆层；可逆性来自连续动力系统的唯一解结构。不过，实际数值求解中仍可能因为误差容差、刚性或反向重构误差导致近似不可逆，所以工程上常结合 checkpointing 或更严格容差。

**误差控制与计算复杂度**

ODE solver 通常根据局部截断误差调节步长。给定容差 $\text{tol}$，求解器试图保证每一步误差在允许范围内。若容差变小，步长会减小，$\operatorname{NFE}$ 增加，结果更精确但更慢；若容差变大，步长会增大，$\operatorname{NFE}$ 降低，结果更快但更粗糙。Neural ODE 的一个特点是，训练和测试可以使用不同容差，以精度换速度。

复杂度上，前向传播主要成本约为 $\operatorname{NFE}_{forward}$ 次神经网络 $f$ 的评估。伴随反向传播需要求解增广系统，单次函数评估会涉及 $f$ 以及向量-Jacobian乘积，因此成本通常是若干倍的 $f$ 评估。内存上，如果不存储完整轨迹，伴随方法相对 solver 内部步数近似为常数内存；如果为了改善数值稳定性使用 checkpointing，则内存和重构精度之间存在折中。

**与离散深度网络的比较**

从表示角度看，ResNet、RevNet、FractalNet、DenseNet 等都可以解释为某种固定步长数值格式的离散化。ResNet 接近 forward Euler，某些多分支或密集连接结构可类比 Runge-Kutta 方法。Neural ODE 的区别不是“用微分方程替代神经网络”，而是把数值积分这一层抽象出来，让神经网络只负责定义连续向量场 $f$，让成熟的 ODE solver 负责沿向量场推进。

这带来一个不同的模型设计哲学。离散网络设计者要决定层数、连接方式和每层计算；Neural ODE 设计者要决定向量场结构、时间依赖形式、容差、正则化和求解器。前者的深度是显式架构超参数，后者的有效深度由数据、动力学复杂度和误差控制共同决定。

**理论主线总结**

Neural ODE 的第一条理论主线是连续深度：从 $h_{k+1}=h_k+f(h_k,\theta_k)$ 到 $\frac{dz}{dt}=f(z,t,\theta)$，残差网络被解释为 ODE 的 Euler 离散化。第二条理论主线是连续反向传播：定义 $a(t)=\frac{\partial L}{\partial z(t)}$，通过 $\frac{da}{dt}=-a^\top\frac{\partial f}{\partial z}$ 反向积分，并通过 $-\int_{t_1}^{t_0}a^\top\frac{\partial f}{\partial\theta}dt$ 得到参数梯度。第三条理论主线是连续变量替换：普通 flow 的 $\log|\det J|$ 在连续极限中变成 $\operatorname{tr}\frac{\partial f}{\partial z}$，从而得到 CNF。

**一句话总结**

Neural ODE 把深度网络从“固定层数的离散变换序列”改写为“由神经网络参数化的连续时间动力系统”：前向传播是求解初值问题，反向传播是求解伴随初值问题，密度建模是同时求解状态和 log-density 的连续变量替换方程，而它的核心数学工具正是 Euler 极限、伴随灵敏度分析、ODE 存在唯一性和 trace 形式的瞬时变量替换公式。
