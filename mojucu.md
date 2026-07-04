---
title: "MuJoCo 论文阅读笔记：Convex and analytically-invertible dynamics with contacts and constraints"
authors: ["Emanuel Todorov"]
type: reading-note
source:
  - "[[mojucu/Convex and analytically-invertible dynamics with contacts and constraints: Theory and implementation in MuJoCo.pdf]]"
  - "尝试下载 TodorovIROS12.pdf，但远端返回的是网页 HTML；本笔记以本地 PDF 为准。"
tags: [MuJoCo, physics-simulation, contact-dynamics, convex-optimization, inverse-dynamics, robotics, optimal-control]
created: 2026-07-02
status: first-pass
---

# MuJoCo 论文阅读笔记

> [!summary]
> 这篇论文的核心不是“又做了一个刚体仿真器”，而是提出一条很明确的动力学建模路线：把接触、关节限制、干摩擦、完整约束这些通常导致非光滑、不可逆、甚至 NP-hard 的对象，改写成带正则化的凸优化问题。这样 forward dynamics 仍然需要迭代求解，但 inverse dynamics 可以被解析反演，并且快一个数量级。MuJoCo 的名字在这里可以理解为 Multi-Joint dynamics with Contact，它服务的重点不是离线动画，而是模型预测控制、轨迹优化、状态估计这类需要“反复调用动力学”的控制问题。

## 1. 论文要解决的问题

机器人真正有用的行为往往发生在接触中：脚踩地、手推墙、物体被夹持、关节到达限位、肌腱或绳索约束运动。困难在于接触事件会把原本光滑的动力系统变成带不等式约束的混合系统。传统思路有两类：第一类是用弹簧阻尼器把接触“抹平”，这样方程光滑、容易反演，但会带来穿透、刚度参数难调和数值不稳定；第二类是 velocity-stepping complementarity，把接触冲量写成互补问题，例如法向冲量非负、分离速度非负、二者乘积为零。这种方法物理含义漂亮，也比显式弹簧稳定，但精确摩擦接触问题会变得计算困难，而且刚体互补模型本身通常不可逆。

论文的关键判断是：如果目标是机器人控制，那么“严格互补”未必是最好的标准。真实接触本来就有材料形变、传感误差和模型误差；在刚体近似里坚持无限硬的接触，反而会让动力学不可逆、不可微、不适合优化。于是作者把接触和约束软化，但这种软化不是朴素的 spring-damper，而是在冲量空间里加正则项，使问题保持凸性，并且使反问题可解析处理。


> [!important] 为什么接触会把光滑 ODE 变成混合系统
> 没有接触时，刚体系统通常可以写成光滑常微分方程：状态 $(q,v)$ 随时间连续演化，力到加速度的映射也是连续的。接触事件则像一个物理开关：两个物体从未接触变为接触，或者从接触变为分离时，系统方程、可行速度集合和允许的力都会突然改变。因此，接触动力学不再只是单一 ODE，而是一个同时包含连续演化和离散模式切换的混合系统。这里的不等式约束来自接触的单边性：距离或间隙必须满足 $d(q)\ge 0$，法向冲量必须满足 $f_n\ge0$，也就是物体不能互相穿透，接触面只能推不能拉。
>
> 这种混合系统的麻烦主要有四个。第一，刚体碰撞可以在极短时间内改变速度，理想模型中甚至表现为速度跳变，因此轨迹对初值和参数可能不可微，这会伤害轨迹优化、MPC 和可微仿真。第二，逆动力学会欠定，典型例子是推墙：如果墙和手都完全刚，只看手的位置和速度无法知道你用了多大的推力，因为一大段不同的推力都会对应同样的零位移。第三，接触模式会组合爆炸；如果有 $N$ 个接触点，每个点都可能处于未接触、粘滞、滑动等状态，模式数会随 $N$ 指数增长。第四，严格互补条件是硬开关，在离散时间数值求解中容易出现接触/分离之间的高频切换，也就是 chattering。

## 2. 基本记号和离散动力学

论文使用广义坐标下的多刚体动力学。令 $q$ 表示关节位置，$v$ 表示关节速度，$u$ 表示控制力，$h>0$ 表示离散时间步长。系统惯性矩阵为 $M(q)$，科氏、离心和重力项合并为 $c(q,v)$，被动弹簧阻尼等项记为 $p(q,v)$。完整约束的 Jacobian 记为 $J_E(q)$，其冲量为 $f_E$；接触、关节限位、干摩擦等非完整或不等式对象的 Jacobian 记为 $J(q)$，其冲量为 $f$。连续形式写作

$$
M\,dv + c\,dt = (p+u)\,dt + J^T f + J_E^T f_E.
$$

这里需要注意一个容易混淆的点：$f$ 和 $f_E$ 是冲量，而不是瞬时力，所以论文没有把它们除以 $dt$。进入离散时间后，令下一步速度为 $v' = v(t+h)$，则有

$$
M(v'-v) = (p+u-c)h + J^T f + J_E^T f_E. \tag{1}
$$

这个式子是整篇论文的骨架。前向动力学的任务是：给定 $q,v,u$，求 $v'$ 以及冲量；逆动力学的任务是：给定 $q,v,v'$，恢复需要的控制力 $u$ 以及相应冲量。真正的难点不在无接触部分，而在如何确定 $f$ 与 $f_E$。


### 2.1 变量表：按论文原文 Notation 补全



| 符号 | 类型或形状 | 原文/物理含义 | 出现位置 |
|---|---:|---|---|
| $q$ | $\mathbb R^{n_q}$ | joint position，广义坐标/关节位置 | Notation, Eq. (1) |
| $v$ | $\mathbb R^{n_v}$ | joint velocity，广义速度/关节速度 | Notation, Eq. (1) |
| $u$ | $\mathbb R^{n_v}$ 或 actuator 映射后的广义力 | control force，控制力/控制广义力 | Notation, Eq. (1) |
| $h$ | 正标量 | discrete timestep，离散仿真步长 | Notation, Eq. (3) |
| $D$ | $\mathbb R^{n_v\times n_v}$ 对角非负 | armature, implicit damping inertia；电机转子惯量与隐式阻尼并入的额外惯性 | Notation, Phase I |
| $M(q)$ | $\mathbb R^{n_v\times n_v}$ 对称正定 | total joint-space inertia，总关节空间惯性；论文中 $M=(M_{CRB}+D)$ | Notation, Eq. (1), (3) |
| $c(q,v)$ | $\mathbb R^{n_v}$ | gravity, Coriolis, centripetal forces；重力、科氏、离心项合并 | Notation, Eq. (1), (2) |
| $p(q,v)$ | $\mathbb R^{n_v}$ | spring-dampers, other passive forces；弹簧阻尼和其他被动力 | Notation, Eq. (1) |
| $J_E(q)$ | $\mathbb R^{m_E\times n_v}$ | equality constraint Jacobian，完整/等式约束 Jacobian | Notation, Eq. (1), (4)-(7) |
| $f_E(q,v,u)$ | $\mathbb R^{m_E}$ | equality constraint impulse，等式约束冲量 | Notation, Eq. (1), (13) |
| $J(q)$ | $\mathbb R^{m\times n_v}$ | contact Jacobian；接触、限位、关节干摩擦等 impulse space 的 Jacobian | Notation, Eq. (1), (8) |
| $f(q,v,u)$ | $\mathbb R^m$ | contact impulse；广义地包括干摩擦、限位、摩擦接触冲量 | Notation, Eq. (1), (8)-(11) |
| $v^-,v^+,v^*$ | $\mathbb R^m$ | impulse-space velocities；$v^-=J\hat v$，$v^+=Jv'$，$v^*$ 是期望下一步接触空间速度 | Notation, Eq. (8), (10)-(11) |
| $\epsilon$ | 正标量 | impulse regularization scaling，冲量正则缩放/软度参数 | Notation, Eq. (26)-(28) |
| $\kappa$ | 正标量，时间量纲 | error reduction time constant，误差衰减时间常数 | Notation, Eq. (28) |
| $\eta$ | 非负标量或向量 | loss due to dry joint friction，干关节摩擦损失阈值 | Notation, Eq. (9), Theorem 4 |
| $d$ 或 $d(i)$ | $\{2,3,5\}$ 等整数 | number of contact friction dimensions；接触摩擦维度，可含滑动、扭转、滚动摩擦 | Notation, Eq. (9) |
| $\mu_1,\ldots,\mu_d$ | 正标量 | coefficients of elliptical friction cone，椭圆摩擦锥各方向摩擦系数 | Notation, Eq. (9), (17), (21)-(22) |
| $R_E$ | $\mathbb R^{m_E\times m_E}$ 对角正定 | equality constraint soft regularizer，软 Gauss 原理中的等式约束正则 | Eq. (5), (7), (13) |
| $a$ | $\mathbb R^{n_v}$ | acceleration；在 Gauss 原理中表示连续时间加速度 | Eq. (4)-(5) |
| $\tau$ | $\mathbb R^{n_v}$ | unconstrained generalized force；Gauss 原理中无约束动力学 $Ma=\tau$ 的右端 | Eq. (4)-(5) |
| $a_E^*$ | $\mathbb R^{m_E}$ | desired equality-constraint acceleration；硬/软 Gauss 原理中的目标约束加速度 | Eq. (4)-(5) |
| $v_E^*$ | $\mathbb R^{m_E}$ | desired next-step velocity in equality-constraint space，由 constraint stabilization 给出 | Theorem 1, Eq. (7), (13) |
| $v'$ 或 $v^0$ | $\mathbb R^{n_v}$ | next-step velocity，论文原文用 $v^0$ 表示 $v(t+h)$；笔记中写作 $v'$ | Eq. (3), (6)-(8), inverse dynamics |
| $\widehat M$ / 原文 $\tilde M$ | $\mathbb R^{n_v\times n_v}$ | apparent inertia taking equality constraints into account；$M+J_E^TR_E^{-1}J_E$ | Eq. (6)-(7) |
| $\hat v$ / 原文 $\tilde v$ | $\mathbb R^{n_v}$ | next-step velocity before contact impulse but after equality-constraint soft effect | Eq. (6)-(7) |
| $A$ | $\mathbb R^{m\times m}$ 半正定 | inverse apparent inertia in contact/impulse space；$A=J\widehat M^{-1}J^T$ | Eq. (8), (10)-(11) |
| $\phi(f)$ | $\mathbb R^{m}$ 或按对象拼接 | inequality left-hand sides；把干摩擦区间、限位半轴、摩擦锥统一写成 $\phi(f)\ge0$ | Eq. (9)-(11) |
| $R$ | $\mathbb R^{m\times m}$ 对角正定 | impulse regularizer；接触冲量正则，保证严格凸并使 inverse dynamics 可解析 | Eq. (10)-(11), (15) |
| $\dot v$ | $\mathbb R^{n_v}$ | inverse dynamics 中由观测速度差得到的加速度，$\dot v=(v'-v)/h$ | Section III-A |
| $\operatorname{RNE}(q,v,\dot v)$ | $\mathbb R^{n_v}$ | Recursive Newton-Euler inverse dynamics 输出；论文中 $\operatorname{RNE}(q,v,\dot v)=(M(q)-D)\dot v+c(q,v)$ | Eq. (2), (12) |
| $y$ | $\mathbb R^m$ | inverse contact projection 的目标点，$y=R^{-1}v^*-v^+$ | Eq. (15) |
| $x$ | 局部接触冲量向量 | friction cone projection 中的待求投影点，$x=(x_0,x_1,\ldots,x_d)$ | Section III-E |
| $C$ | 凸锥集合 | elliptical friction cone，$x_0\ge0,\ x_0^2\ge\sum_j\mu_j^{-2}x_j^2$ | Section III-E |
| $r_k$ | 正标量 | $R$ 的局部对角权重，weighted cone projection 的距离权重 | Section III-E |
| $\hat x,\hat y$ | 变换后向量 | $\hat x_k=r_k^{1/2}x_k$，$\hat y_k=r_k^{1/2}y_k$，把带权距离变成欧氏距离 | Eq. (16) |
| $\hat\mu$ | 正标量 | transformed circular cone coefficient，使椭圆锥在变换后变成圆锥 | Eq. (17)-(22) |
| $r_0,r_F$ | 正标量 | normal regularizer 与 mean friction regularizer，用户指定的接触正则参数 | Eq. (21)-(22) |
| $B,K$ | 对角或矩阵 | stabilization damping/stiffness；Baumgarte 风格二阶误差稳定器 | Eq. (24), (28) |
| $\mathcal J$ / 原文堆叠 $J$ | 矩阵 | Section V 中把 $J_E$ 和 $J$ 纵向堆叠后的 combined constraint/contact Jacobian | Section V |
| $x$（Section V） | 约束/接触空间坐标 | 对约束、限位、接触法向表示 violation/penetration distance；对摩擦维表示相对某个任意 offset 的坐标 | Section V |
| $\dot x,\dot x^*$ | 约束/接触空间速度 | $\dot x=\mathcal Jv$，$\dot x^*$ 是稳定器给出的期望下一步速度 | Eq. (24) |
| $a$（Section V） | $\mathbb R^{m_E+m}$ | 非冲击力在 combined space 中造成的加速度，$a=\mathcal JM^{-1}(p-c+u)$ | Section V |

> [!note] 原文符号与本笔记符号的对应
> 原文把下一步速度写成 $v^0$，本笔记为了避免和指数混淆写成 $v'$；原文 Theorem 1 中的 $\tilde M,\tilde v$ 在本笔记中写成 $\widehat M,\hat v$。这两个替换只是排版选择，含义不变。

### 2.1.1 公式缺漏核对表

| 原文编号 | 原文内容 | 当前笔记处理 | 是否已补齐 |
|---|---|---|---|
| Eq. (1) | 连续时间含冲量动力学 $Mdv+c\,dt=(p+u)dt+J^Tf+J_E^Tf_E$ | 第 2 节已有 | 已有 |
| Eq. (2) | $\operatorname{RNE}(q,v,\dot v)=(M(q)-D)\dot v+c(q,v)$ | 原笔记缺少完整说明 | 本表与变量表补齐；建议后文逆动力学处引用 |
| Eq. (3) | 离散动力学 $M(v'-v)=(p+u-c)h+J^Tf+J_E^Tf_E$ | 第 2 节已有 | 已有 |
| Eq. (4) | 硬 Gauss 原理 | 第 3 节已有，编号在笔记中为式 (2) | 已有但编号不同 |
| Eq. (5) | 软 Gauss 原理 | 第 3 节已有，编号在笔记中为式 (3) | 已有但编号不同 |
| Eq. (6)-(7) | $v'=\hat v+\widehat M^{-1}J^Tf$，$\widehat M,\hat v$ 定义 | 第 3 节已有 | 已有 |
| Eq. (8) | 接触空间动力学 $Af+v^-=v^+$ | 第 4 节已有 | 已有 |
| Eq. (9) | friction / limit / contact 三类不等式 | 第 4 节有摩擦锥，但干摩擦双边不等式和 limit 原文形式需要保留 | 变量表已补；正文可再强化 |
| Eq. (10)-(11) | complementarity-free forward contact objective 及代入后 QP | 第 5 节已有 | 已有 |
| Eq. (12)-(13) | inverse dynamics 的 RNE 方程与 $J_E^Tf_E$ 恢复 | 原笔记有逆动力学直觉，但缺 Eq. (12) 完整式 | 本表标出，后续应补入第 6 节 |
| Eq. (14a)-(14b) | 抽象凸优化反演定理 Theorem 3 | 原笔记未写 | 若只做主线笔记可略；若做完整推导应补 |
| Eq. (15) | inverse contact least squares projection | 第 6 节已有，编号为式 (10) | 已有但编号不同 |
| Eq. (16)-(22) | 摩擦锥投影变换、圆锥条件、解析投影、正则参数构造 | 第 7 节部分已有，但公式 (18)-(22) 不完整 | 需要补充或标注“略去实现细节” |
| Eq. (23)-(28) | impulse dynamics 参数分析与 $\epsilon,\kappa$ 调参 | 第 8 节部分已有 | 需要补齐 $R_{ii}$、$B_{ii}$、$K_{ii}$ 原文公式 |

### 2.2 物理概念速查

**广义坐标与广义速度。** 广义坐标 $q$ 是描述机器人构型的最小或近似最小变量，例如一个转动关节用角度，一个滑动关节用位移，自由刚体可以用位置加四元数。广义速度 $v$ 是构型变化率在速度空间中的表示。对普通关节系统，可以近似把 $v$ 看成 $\dot q$；但在包含三维姿态四元数时，$q$ 和 $v$ 的维度与几何含义并不完全相同。

**质量矩阵 $M$。** 质量矩阵不是简单的“每个关节的质量”，而是系统动能在广义速度坐标下的二次型。动能可以写成

$$
T=\frac12v^TM(q)v.
$$

因此 $M$ 定义了“改变某个速度方向有多难”。Gauss 原理里的距离使用 $M$ 加权，正是因为不同方向的加速度改变对应不同动能代价。


**Gauss 最小约束原理。** 这个原理可以理解成“约束力只做最小必要修改”。先想象没有任何约束时，系统在外力作用下会产生一个无约束加速度 $a_0$。如果无约束动力学写成

$$
Ma_0=\tau,
$$

那么 $a_0=M^{-1}\tau$。但是系统如果有完整约束，例如杆长固定、铰链闭合、点必须留在某个曲面上，真实加速度 $a$ 不能随便取，而必须满足约束在加速度层面的条件

$$
J_Ea=a_E^*.
$$

Gauss 原理说，真实加速度不是任意一个满足约束的加速度，而是在所有满足 $J_Ea=a_E^*$ 的候选加速度里，最接近无约束加速度 $a_0$ 的那个。这个“接近”用质量矩阵 $M$ 度量：

$$
a
=
\arg\min_{\bar a:\,J_E\bar a=a_E^*}
\frac12(\bar a-a_0)^TM(\bar a-a_0).
$$

把 $a_0=M^{-1}\tau$ 代入，也可以写成论文里的形式：

$$
\min_{a:\,J_Ea=a_E^*}(Ma-\tau)^TM^{-1}(Ma-\tau).
$$

这两个式子是同一个意思。第一种写法强调“加速度离无约束加速度最近”；第二种写法强调“实际广义力 $Ma$ 离无约束广义力 $\tau$ 最近”。质量矩阵出现在这里不是装饰，而是因为改变不同方向的加速度需要付出不同的动能代价。一个很重的方向上改一点加速度，代价比一个很轻的方向大。

从拉格朗日乘子的角度看，Gauss 原理等价于普通约束动力学。对上面的约束优化写一阶条件，会得到

$$
M(a-a_0)+J_E^T\lambda=0.
$$

由于 $Ma_0=\tau$，所以

$$
Ma=\tau-J_E^T\lambda.
$$

如果把乘子符号反过来定义，就得到常见形式

$$
Ma=\tau+J_E^T\lambda.
$$

这里的 $J_E^T\lambda$ 就是约束力。因此，Gauss 原理并不是另一套神秘力学，而是把“求约束力”解释为一个投影问题：先看外力本来会让系统怎么加速，再用最小改动把这个加速度投影回满足约束的集合。MuJoCo 的“软 Gauss 原理”就是把这个硬投影变成软惩罚；约束不再必须完全满足 $J_Ea=a_E^*$，而是违反约束会被 $R_E^{-1}$ 加权惩罚，从而让动力学可逆、数值更稳定。

**Bias force $c(q,v)$。** 多刚体系统里，即使没有控制力，也会有由重力、科氏力、离心力导致的加速度趋势。论文把这些项合并成 $c(q,v)$。如果把动力学写成更常见的形式，通常是

$$
M(q)\dot v+c(q,v)=p(q,v)+u+J^Tf+J_E^Tf_E.
$$

所以 $c$ 在离散式里被移到右边后变成 $(p+u-c)h$。

**Jacobian 的物理意义。** 约束 Jacobian $J$ 把广义速度映射成某个接触点或约束方向上的速度：

$$
v_{contact}=Jv.
$$

它的转置 $J^T$ 则把接触空间中的力或冲量映射回广义坐标。这个转置关系来自虚功原理：如果接触空间冲量是 $f$，虚位移导致的接触位移是 $J\delta q$，那么虚功为 $f^TJ\delta q=(J^Tf)^T\delta q$，所以广义冲量就是 $J^Tf$。

**完整约束。** 完整约束是位置层面的约束，例如两个刚体用铰链连接、一个点必须留在某个面上、闭链机构的末端必须重合。它可以写成 $\phi(q,t)=0$。MuJoCo 对这类约束不是用无限大的约束力强行满足，而是通过 $R_E$ 软化以后吸收到 $\widehat M$ 和 $\hat v$ 中。

**接触约束。** 接触约束通常是不等式：物体不能穿透地面，法向冲量不能拉住物体，只能推开物体。因此法向接触常有 $f_n\ge0$。如果还有库仑摩擦，切向冲量不能超过法向冲量乘以摩擦系数，即 $\|f_t\|\le \mu f_n$。这就是摩擦锥。

**冲量正则项 $R$。** 真实世界的接触不是无限硬的，材料会形变，接触面会有微小压缩。刚体模型把这些细节全丢掉了。MuJoCo 的 $R$ 可以理解为把这些未建模形变压缩成一个冲量空间里的软度参数。数学上，$R$ 保证优化问题严格凸；物理上，它避免“完全刚体接触”导致的不可逆和不唯一。

**表观惯性 $A$。** 接触空间矩阵 $A=J\widehat M^{-1}J^T$ 描述“在某个接触方向施加单位冲量，会让接触速度改变多少”。如果 $A$ 的某个方向小，说明这个方向上系统显得很重，单位冲量改变速度很少；如果大，说明系统显得轻。多接触时，$A$ 的非对角项描述接触之间的耦合。

**前向动力学与逆动力学。** 前向动力学问：给定当前状态和控制力，下一步怎么动？在这篇论文里就是给定 $q,v,u$ 求 $v'$ 和冲量。逆动力学问：如果我观察到系统从 $v$ 变到 $v'$，那需要什么控制力和接触冲量？控制、轨迹优化和状态估计经常需要逆动力学，因为它可以判断某条运动是否物理合理。

**严格互补与软化接触。** 传统刚体接触喜欢写 $f_n\ge0$、$v_n^+\ge0$、$f_nv_n^+=0$。这表达“有冲量就不分离，分离就没有冲量”。但它会让问题变成非光滑互补问题，摩擦下尤其麻烦。MuJoCo 的策略是放弃严格互补，改用凸优化和正则化，让模型更适合数值求解、控制和反演。

## 3. 软化 Gauss 原理：完整约束如何进入动力学


> [!note] 先解释“完整约束”和 Gauss 最小约束原理
> **完整约束**，也就是 holonomic constraint，指的是约束可以写成位置和时间的方程
>
> $$
> \phi(q,t)=0.
> $$
>
> 例如一个质点被限制在半径为 $R$ 的圆上运动，约束就是 $x^2+y^2-R^2=0$。这种约束直接限制构型 $q$，所以叫完整约束。与之相对，某些轮式机器人“不侧滑”的限制只能自然写成速度约束，通常不能积分成纯位置方程，那类约束叫非完整约束。
>
> **Gauss 最小约束原理**说：如果没有约束，外力会让系统产生某个无约束加速度；但约束存在时，真实加速度必须满足约束的加速度条件。真实加速度是在所有满足约束的候选加速度里，离无约束加速度最近的那个。这里的“最近”不是普通欧氏距离，而是由质量矩阵 $M$ 加权的距离。

更具体地说，假设广义坐标是 $q$，质量矩阵是 $M(q)$，广义力是 $Q$。如果没有约束，系统满足

$$
M(q)\ddot q_0=Q,
$$

所以无约束加速度是

$$
\ddot q_0=M(q)^{-1}Q.
$$

如果系统还满足完整约束 $\phi(q,t)=0$，那么对约束连续求两次时间导数，可以得到加速度层面的线性约束

$$
A(q,t)\ddot q=b(q,\dot q,t),
$$

其中 $A(q,t)=\frac{\partial \phi}{\partial q}$，而 $b(q,\dot q,t)$ 收集了由 $\dot q$、显式时间项和二阶导数带来的项。于是 Gauss 原理可以写成一个约束最小二乘问题：

$$
\ddot q
=
\arg\min_{a}\frac12(a-\ddot q_0)^TM(q)(a-\ddot q_0)
\quad\text{s.t.}\quad
A(q,t)a=b(q,\dot q,t).
$$

它的直觉非常朴素：约束力不应该主动做多余的事情。外力本来想让系统产生 $\ddot q_0$，但这个加速度可能违反约束；约束力只做最小必要修改，把加速度投影到满足约束的集合上。这个投影使用质量矩阵作为度量，所以重的方向和轻的方向“改起来代价不同”。

这个最小化问题和拉格朗日乘子法是等价的。写出 Lagrangian

$$
\mathcal L(a,\lambda)=\frac12(a-\ddot q_0)^TM(a-\ddot q_0)+\lambda^T(Aa-b),
$$

对 $a$ 求一阶条件得到

$$
M(a-\ddot q_0)+A^T\lambda=0.
$$

由于 $M\ddot q_0=Q$，把真实加速度记为 $\ddot q$，可得

$$
M\ddot q=Q-A^T\lambda.
$$

如果把乘子符号换成 $-\lambda$，就得到更常见的写法

$$
M\ddot q=Q+A^T\lambda.
$$

其中 $A^T\lambda$ 正是约束力。也就是说，Gauss 最小约束原理不是另一套不同的力学，而是把“求约束力”改写成“在满足约束的加速度集合上做质量加权投影”。MuJoCo 后面的软化版本，就是把这个硬投影改成带惩罚项的软投影。

对于完整约束，经典 Gauss 最小约束原理可以这样理解：在所有满足约束加速度的候选加速度中，真实加速度选择与无约束加速度“最接近”的那个。若无约束动力学为 $Ma=\tau$，约束为 $J_E a = a_E^*$，那么硬约束版本是

$$
\min_{a:\,J_Ea=a_E^*} (Ma-\tau)^T M^{-1}(Ma-\tau). \tag{2}
$$

这当然符合刚体约束直觉，但它的问题是太硬：一旦只观察到运动学状态，约束冲量不一定能被唯一恢复。MuJoCo 在这里做的第一步是把硬等式约束变为软惩罚：

$$
\min_a (Ma-\tau)^T M^{-1}(Ma-\tau)
+ (J_Ea-a_E^*)^T R_E^{-1}(J_Ea-a_E^*), \tag{3}
$$

其中 $R_E$ 是正定对角正则矩阵。直觉上，$R_E$ 越小，约束越硬；$R_E$ 越大，约束越软。这个式子的重要性在于它仍然是凸二次问题，并且有解析解。离散化以后，论文得到

$$
v' = \hat v + \widehat M^{-1}J^T f, \tag{4}
$$

其中

$$
\widehat M = M + J_E^T R_E^{-1}J_E,
$$

并且

$$
\hat v = v + \widehat M^{-1}\big((p+u-c)h + J_E^T R_E^{-1}(v_E^* - J_Ev)\big). \tag{5}
$$

这组式子的含义很值得停一下看。完整约束并没有被当作一个必须额外求出的未知冲量，而是先被吸收到一个“约束修正后的表观惯性” $\widehat M$ 和一个“接触前速度” $\hat v$ 中。换句话说，在求接触冲量之前，MuJoCo 先把完整约束对运动的影响折叠进了等效动力学。

## 4. 接触空间：把动力学投影成一个冲量问题

接触冲量只通过 $J^Tf$ 作用在广义坐标中，因此可以把式 (4) 乘以 $J$，得到接触空间中的动力学：

$$
Jv' = J\hat v + J\widehat M^{-1}J^T f.
$$

论文把它记成更紧凑的形式：

$$
Af + v^- = v^+, \tag{6}
$$

其中

$$
A = J\widehat M^{-1}J^T,\qquad v^- = J\hat v,\qquad v^+ = Jv'.
$$

这里的 $A$ 是接触空间里的 inverse apparent inertia。可以把它想象成“单位接触冲量会在接触速度上造成多大变化”。如果有多个接触点，$A$ 会把它们耦合起来：一个脚的冲量可能改变身体姿态，从而影响另一个接触点的速度。这就是为什么接触动力学不能简单地逐点处理。

接下来需要给 $f$ 加物理约束。论文统一处理了三类对象：干摩擦、限位和摩擦接触。干摩擦满足 $-\eta \le f_i \le \eta$；限位和无摩擦法向接触满足 $f_i\ge 0$；摩擦接触还要求冲量落在椭圆摩擦锥中。对于一个接触，若 $f_i$ 是法向冲量，$f_{i+1},\ldots,f_{i+d}$ 是切向、扭转或滚动摩擦冲量，则约束写成

$$
f_i \ge 0,\qquad
f_i^2 - \sum_{j=1}^{d}\left(\frac{f_{i+j}}{\mu_j}\right)^2 \ge 0. \tag{7}
$$

当 $d=2$ 且两个摩擦系数相同为 $\mu$ 时，式 (7) 就退化为熟悉的库仑摩擦锥：

$$
\mu f_i \ge \sqrt{f_{i+1}^2+f_{i+2}^2}.
$$

## 5. 为什么不用严格互补，而用 complementarity-free 接触

传统互补接触会给法向接触加上类似条件：法向冲量 $f_i\ge 0$，下一步法向速度 $v_i^+\ge 0$，并且 $f_i v_i^+=0$。这句话的物理意思是：要么接触点分离，没有冲量；要么存在接触冲量，法向速度为零。问题是摩擦接触下这类互补问题计算困难，而且从逆动力学角度看不可逆。推墙就是最简单的反例：如果墙完全刚，手的位置和速度不包含墙内部形变信息，那么仅凭运动学无法知道你用了多大力推墙。



> [!note] complementarity-free 的核心转向
> 传统严格互补方法把接触写成“非黑即白”的逻辑：法向冲量 $f_i\ge0$，分离速度 $v_i^+\ge0$，并且 $f_iv_i^+=0$。这表示要么分离且没有冲量，要么接触且法向速度被压到边界上。这个模型很符合刚体接触的理想图像，但它在干摩擦、多接触和逆动力学中会变得难解、不可逆且不稳定。MuJoCo 的核心转向是移除严格互补，把接触冲量的求解改写成带正则项的凸优化问题。这样接触不再是一个硬到不可逆的离散开关，而是一个具有可调软度的连续投影问题。

MuJoCo 的替代方案是直接定义冲量 $f$ 为一个凸优化问题的解。它最小化“相对于期望接触速度 $v^*$ 的下一步动能偏差”加“冲量正则项”：

$$
\min_{\phi(f)\ge 0}
(v^+-v^*)^T A^{-1}(v^+-v^*) + f^T Rf. \tag{8}
$$

把 $v^+=Af+v^-$ 代入后，得到前向接触动力学实际求解的问题：

$$
\min_{\phi(f)\ge 0}
\frac12 f^T(A+R)f + f^T(v^- - v^*). \tag{9}
$$

这是全篇最核心的式子之一。$A$ 负责接触之间的动力学耦合，$R$ 是正定对角正则项，$\phi(f)\ge 0$ 表示干摩擦、限位、摩擦锥等不等式。由于 $A$ 半正定且 $R$ 正定，目标函数严格凸，因此解唯一。严格互补被拿掉后，接触不再是“硬到不可逆”的理想开关，而是一个可调软度的凸投影问题。

## 6. 逆动力学为什么会变简单

逆动力学给定的是 $q,v,v'$，所以 $v^+=Jv'$ 已知。论文的漂亮之处在于：前向问题虽然依赖 $A$，但反向问题在特殊正则形式下可以分解成许多小的投影问题。对于式 (9) 对应的正则项，逆接触冲量可以写成

$$
\min_{\phi(f)\ge 0}\frac12(f-y)^T R(f-y), \tag{10}
$$

其中

$$
y = R^{-1}v^* - v^+. \tag{11}
$$

这意味着逆接触动力学不再需要构造和因式分解完整的 $A=J\widehat M^{-1}J^T$。它只需要在由干摩擦区间、限位半轴和摩擦锥构成的可行集合上，对向量 $y$ 做带权投影。更重要的是，因为 $R$ 是对角的，不同接触对象之间在逆问题中解耦。

从数学上看，这是一种非常有用的“不对称设计”：前向动力学保留 $A$，所以它能感知多接触之间的耦合；逆动力学通过正则化结构把问题拆开，因此非常快。论文的结论部分也强调，反演之所以可能，是因为模型允许一定软度；这个软度不是副作用，而是为了可控、可估计、可优化而主动引入的结构。

## 7. 摩擦锥投影的数学图像

逆接触中最复杂的小问题是把一个向量投影到椭圆摩擦锥。设一个接触的局部冲量为 $x=(x_0,x_1,\ldots,x_d)$，其中 $x_0$ 是法向冲量，后面的分量是摩擦冲量。椭圆锥为

$$
C=\left\{x\in\mathbb R^{d+1}:x_0\ge 0,\ x_0^2\ge \sum_{j=1}^{d}\mu_j^{-2}x_j^2\right\}. \tag{12}
$$

要解的问题是

$$
\min_{x\in C}\sum_{k=0}^{d}r_k(x_k-y_k)^2, \tag{13}
$$

其中 $r_k>0$ 是 $R$ 的对角元素。这个问题本来是“椭圆锥 + 带权距离”，一般不够简单。论文施加一个关键条件

$$
\mu_j^2 r_j = \hat\mu^2 r_0, \tag{14}
$$

于是通过变量替换

$$
\hat x_k = \sqrt{r_k}x_k,\qquad \hat y_k = \sqrt{r_k}y_k,
$$

带权距离变成普通欧氏距离，椭圆锥也变成圆锥。于是 ConeProject 可以解析完成：如果 $\hat y$ 已在锥内，投影就是它自己；如果 $\hat y$ 在锥轴负方向，投影可能是原点；其余情况投影落在锥面上，可以用一个 Lagrange multiplier $\lambda$ 解出。这里的思想比具体公式更重要：MuJoCo 不是任意选择正则项，而是选择一种让几何投影有闭式解的正则项。

> [!note]
> 这也是为什么论文强调 $R$ 要对角、摩擦维度的正则权重要按摩擦系数调整。看起来是实现细节，实际上是让逆动力学从一个耦合二次规划退化为一组解析投影的关键。

## 8. 参数 $\epsilon$ 与 $\kappa$：软度不是随便调的弹簧

论文把冲量动力学的软度压缩到两个主要参数：冲量正则缩放 $\epsilon$ 和误差衰减时间常数 $\kappa$。在接触或约束空间中，用 $x$ 表示约束违反量或穿透量，用 $\dot x=Jv$ 表示相应速度。期望下一步速度通过类似 Baumgarte stabilization 的二阶机制设置：

$$
\dot x^* = \dot x - hB\dot x - hKx. \tag{15}
$$

在非滑动、约束不活跃切换的理想分析里，论文得到接触/约束空间动力学近似为

$$
\ddot x + 2\kappa^{-1}\dot x + \kappa^{-2}x \approx \frac{\epsilon}{1+
\epsilon}a. \tag{16}
$$

这说明参数的意义很清楚。$\kappa$ 是误差恢复的时间尺度；越小，系统越快消除穿透和约束误差。$\epsilon$ 控制接触有多接近硬约束；当 $\epsilon\to 0$ 时，右侧趋近于零，接触约束几乎自主消除误差；当 $\epsilon$ 较大时，外部加速度更明显地影响接触违反量。对于一个物体在重力 $g$ 下压在地面上的简单情形，稳态穿透近似为

$$
x \approx \frac{\kappa^2\epsilon g}{1+
\epsilon}. \tag{17}
$$

这个公式很有解释力：穿透深度和质量无关，而由重力、误差时间常数和软度控制。这比普通 spring-damper 的调参更“动力学无关”，因为刚度和阻尼会随惯性自动缩放。

## 9. 复杂度和实验结论

前向动力学要计算并分解惯性矩阵，还要构造接触空间矩阵 $A=J\widehat M^{-1}J^T$，再用 GPGS 这类迭代方法解凸优化问题。因此最坏复杂度中会出现 $O(n^3)$ 和与接触维度 $m$ 有关的二次项。论文强调 MuJoCo 利用机器人树结构的稀疏性，实际表现会好于朴素最坏界。

逆动力学的复杂度更低。它主要使用 RNE 计算惯性项，然后用解析冲量投影处理接触；不需要构造完整 $A$。论文实验中，27 自由度 humanoid 在大量接触冲量下，逆动力学可在几十微秒量级完成；前向动力学虽然慢一个数量级，但仍远快于实时。这里的意义不只是“跑得快”，而是“快到可以被优化器在内层反复调用”。这正是 MuJoCo 对模型预测控制、轨迹优化和物理一致状态估计有吸引力的原因。

## 10. 我的理解：这篇论文的核心贡献

第一，论文把接触动力学从互补问题转成带正则化的凸优化问题。这样做牺牲了理想刚体接触的绝对硬度，但换来了唯一解、数值稳定和可反演性。第二，论文把完整约束和接触约束放进同一个冲量空间语言中：完整约束通过软 Gauss 原理进入 $\widehat M$ 和 $\hat v$，接触通过凸投影确定 $f$。第三，论文特别重视逆动力学，因为控制和估计往往不是只问“给定力会怎么动”，还会问“观察到这样运动，需要什么力或冲量”。第四，MuJoCo 的快不是单纯工程优化，而是数学建模选择导致的：正则矩阵 $R$ 的对角结构、摩擦锥投影的闭式解、逆问题中避免构造 $A$，这些都直接服务于速度。

如果用一句话概括：MuJoCo 这篇论文把接触仿真设计成“适合控制的物理近似”，而不是“尽可能硬的刚体真实主义”。它的哲学是，接触模型本来就是现象学模型；与其坚持一个不可逆、不可优化、计算昂贵的理想化互补模型，不如引入可解释的软度，让模型在仿真、控制、估计之间形成统一接口。

## 11. 可复习的公式索引

| 主题 | 公式 | 作用 |
|---|---|---|
| 离散动力学 | $M(v'-v)=(p+u-c)h+J^Tf+J_E^Tf_E$ | 全文动力学主方程 |
| 软 Gauss 原理 | 式 (3) | 把完整约束变成可逆凸二次惩罚 |
| 接触空间动力学 | $Af+v^-=v^+$ | 把广义坐标动力学投影到接触空间 |
| 摩擦锥 | $f_i^2\ge\sum_j(f_{i+j}/\mu_j)^2$ | 表示法向冲量与摩擦冲量的可行集合 |
| 前向接触 | $\min \frac12f^T(A+R)f+f^T(v^- - v^*)$ | 求唯一接触冲量，但通常要迭代 |
| 逆接触 | $\min \frac12(f-y)^TR(f-y)$ | 变成解析投影问题 |
| 稳定化 | $\dot x^*=\dot x-hB\dot x-hKx$ | 用二阶误差衰减设置期望接触速度 |
| 穿透近似 | $x\approx\kappa^2\epsilon g/(1+\epsilon)$ | 解释软度和误差时间常数的物理效果 |

## 12. 延伸问题

- 如果把 $R$ 取得太小，模型更接近硬接触，但前向求解会更困难，逆动力学也会更敏感；这在控制里可能表现为优化器梯度变坏。
- 严格互补模型并不是“真物理”，因为刚体接触忽略了真实材料形变；MuJoCo 的软化可以看作把未建模形变压缩到冲量正则项中。
- 论文的逆动力学适合做物理一致先验：给定观测到的运动，快速判断需要的接触冲量和控制力是否合理。
- 这篇论文解释了为什么 MuJoCo 在强化学习和最优控制中长期流行：它不仅快，而且动力学形式对优化友好。

## 13. 原文公式与附录核对补记

> [!check]
> 已核对 `/home/zihan/书库/Control/mojucu/Convex and analytically-invertible dynamics with contacts and constraints: Theory and implementation in MuJoCo.pdf`。该 PDF 共 8 页，正文从 Abstract 到 References 结束，没有单独的 Appendix/附录。因此本地原文不存在“附录公式”可补；需要补的是正文中原笔记为了阅读流畅而省略的若干公式细节，主要集中在 Theorem 3、摩擦锥解析投影、Algorithm 5 和 Section V 的参数调节部分。

### 13.1 原文编号公式核对结论

原文编号公式从 Eq. (1) 到 Eq. (28)，其中 Eq. (14) 分为 Eq. (14a) 与 Eq. (14b)。前面的阅读笔记已经覆盖了核心主线：连续与离散动力学、软 Gauss 原理、接触空间方程、摩擦锥约束、前向凸优化、逆向最小二乘投影、稳定化参数 $\epsilon$ 与 $\kappa$ 的物理含义。真正有缺漏的是四组公式：第一，Eq. (12)-(13) 的逆动力学总方程和完整约束冲量恢复公式没有在正文中充分展开；第二，Eq. (14a)-(14b) 的抽象反演定理没有写出；第三，Eq. (18)-(22) 的摩擦锥投影闭式解和正则参数构造不完整；第四，Eq. (23)、Eq. (25)、Eq. (27)-(28) 以及接近精确临界阻尼的矩阵方程没有完整保留。下面按原文顺序补齐。

### 13.2 逆动力学总方程：原文 Eq. (12)-(13)

逆动力学部分的输入是 $(q,v,v')$，因此可以先定义离散加速度

$$
\dot v \equiv \frac{v'-v}{h}.
$$

原文用 RNE 算法把惯性项、科氏项和重力项整理出来，得到

$$
\operatorname{RNE}(q,v,\dot v)+D\dot v
= p+u+\frac{J^Tf+J_E^Tf_E}{h}. \tag{原文 12}
$$

这个式子的作用是：只要我们恢复出接触冲量 $f$ 和完整约束冲量在广义坐标中的作用 $J_E^Tf_E$，控制力 $u$ 就可以直接由上式解出。注意原文特别强调，逆动力学不需要计算或分解惯性矩阵 $M$，后面的冲量恢复也不需要完整的 $M$。

对于完整约束，前向动力学中没有显式求 $f_E$，而是把它吸收到 $\widehat M$ 和 $\hat v$ 中；但逆动力学里 $v'$ 已知，所以可以恢复约束冲量的广义力形式：

$$
J_E^Tf_E=J_E^TR_E^{-1}(v_E^*-J_Ev'). \tag{原文 13}
$$

这里要注意，原文 Eq. (13) 恢复的是 $J_E^Tf_E$，不是一定恢复 $f_E$ 本身。如果 $J_E$ 满秩、约束非冗余，那么可以进一步恢复 $f_E$；但逆动力学方程只需要 $J_E^Tf_E$，所以无需这个额外假设。

### 13.3 Theorem 3：一般凸优化反演定理，原文 Eq. (14a)-(14b)

原文 Theorem 3 是逆接触动力学成立的抽象数学理由。设 $A\in\mathbb R^{n\times n}$ 对称半正定，$r,s:\mathbb R^n\to\mathbb R$ 为凸函数，$\phi:\mathbb R^n\to\mathbb R^n$ 表示凸不等式约束，且 $b,c\in\mathbb R^n$。定义 $x_b^*$ 和 $x_c^*$ 分别为下面两个凸优化问题的唯一全局解：

$$
x_b^*:
\quad
\min_{x:\,\phi(x)\ge 0}
 r(x)+s(Ax+b)+\frac12x^TAx+x^Tb, \tag{原文 14a}
$$

以及

$$
x_c^*:
\quad
\min_{x:\,\phi(x)\ge 0}
 r(x)+x^T\big(A\nabla s(c)+c\big). \tag{原文 14b}
$$

定理说，如果 $Ax_b^*+b=c$，那么 $x_b^*=x_c^*$。证明思路是比较两个问题的 KKT 条件。第一个问题的一阶条件包含 $A\nabla s(Ax+b)+Ax+b$，第二个问题的一阶条件包含 $A\nabla s(c)+c$；当 $Ax+b=c$ 时，两套 KKT 条件相同，因此解相同。把这个抽象定理应用到接触动力学时，可以令 $x=f$，$b=v^-$，$c=v^+$，并把 $-f^Tv^*$ 吸收到 $r(f)$ 中。它解释了为什么“给定前向速度 $v^-$ 求 $f$”和“给定后向速度 $v^+$ 反求 $f$”可以得到同一个冲量。

### 13.4 逆接触的解析特殊情形：原文 Eq. (15) 与 Theorem 4

对于原文 Eq. (11) 所用的正则形式，冲量正则项可写成

$$
r(f)=\frac12 f^TRf-f^Tv^*.
$$

因此逆问题化为带约束的带权最小二乘投影：

$$
\min_{f:\,\phi(f)\ge 0}\frac12(f-y)^TR(f-y), \tag{原文 15}
$$

其中

$$
y\equiv R^{-1}v^*-v^+.
$$

Theorem 4 给出逐对象的解析解。对干摩擦对象，解是区间截断：

$$
f_i=\max\big(-\eta(i),\min(\eta(i),y_i)\big).
$$

对限位或无摩擦法向对象，解是半轴投影：

$$
f_i=\max(0,y_i).
$$

对摩擦接触对象，解是把局部向量 $(y_i,\\ldots,y_{i+d(i)})$ 投影到对应摩擦锥中：

$$
f_i,\ldots,f_{i+d(i)}=\operatorname{ConeProject}(y,i).
$$

### 13.5 摩擦锥投影细节：原文 Eq. (16)-(20)

局部摩擦锥投影问题是

$$
C=\left\{x\in\mathbb R^{d+1}:x_0\ge 0,\ x_0^2\ge\sum_{j=1}^{d}\mu_j^{-2}x_j^2\right\},
$$

并要求解

$$
\min_{x\in C}\sum_{k=0}^{d}r_k(x_k-y_k)^2.
$$

为了把带权距离变成普通欧氏距离，原文引入变量替换

$$
\hat x_k\equiv r_k^{1/2}x_k,
\qquad
\hat y_k\equiv r_k^{1/2}y_k. \tag{原文 16}
$$

变换后的锥为

$$
\hat C=\left\{\hat x\in\mathbb R^{d+1}:\hat x_0\ge 0,
\ r_0^{-1}\hat x_0^2\ge\sum_j\mu_j^{-2}r_j^{-1}\hat x_j^2\right\}.
$$

为了让这个锥成为圆锥，原文要求存在标量 $\hat\mu$，使得

$$
\mu_j^2r_j=\hat\mu^2r_0. \tag{原文 17}
$$

这样变换后的锥可写成

$$
\hat C=\left\{\hat x\in\mathbb R^{d+1}:\hat x_0\ge 0,
\ \hat\mu^2\hat x_0^2\ge\sum_j\hat x_j^2\right\}.
$$

如果 $\hat y\in\hat C$，投影就是 $\hat x=\hat y$。否则投影落在锥面上，存在 Lagrange multiplier $\lambda$，满足

$$
\hat x-\hat y+\lambda(\hat\mu^2\hat x_0,-\hat x_1,\ldots,-\hat x_d)^T=0.
$$

由此可得

$$
\hat x_0=\frac{\hat y_0}{1+\hat\mu^2\lambda},
\qquad
\hat x_j=\frac{\hat y_j}{1-\lambda}. \tag{原文 18}
$$

把式 (18) 代回锥面等式，会得到两个候选解对应的乘子：

$$
\lambda^{\pm}=\frac{1\mp\beta}{1\pm\hat\mu^2\beta},
\qquad
\beta=rac{\left(\sum_j\hat y_j^2\right)^{1/2}}{\hat\mu\,\hat y_0}. \tag{原文 19}
$$

原文还处理两个特殊情形。若 $\hat y_0=0$，则通用公式会出现除零，此时解为

$$
\hat x_0=\frac{\hat\mu}{1+\hat\mu^2}\left(\sum_j\hat y_j^2\right)^{1/2},
\qquad
\hat x_j=\frac{\hat\mu^2}{1+\hat\mu^2}\hat y_j. \tag{原文 20}
$$

若所有摩擦方向分量都为零，即 $\hat y_j=0$ 对所有 $j\ge1$ 成立，则最优解是锥尖 $\hat x=0$。一般情形下，原文的选择规则是：由 Eq. (18)-(19) 得到两个候选 $\hat x^\pm$；如果两个候选的第一分量都非负，就选择离 $\hat y$ 更近的那个；如果只有一个候选第一分量非负，就选它；如果两个候选第一分量都为负，就选原点。

### 13.6 正则参数构造与 Algorithm 5：原文 Eq. (21)-(22)

为了满足圆锥化条件 Eq. (17)，原文不是让用户直接指定每个 $r_j$，而是让用户指定摩擦系数、法向正则和平均摩擦正则：

$$
\text{friction coefficients: }\mu_1,\ldots,\mu_d,
\qquad
\text{normal regularizer: }r_0,
\qquad
\text{mean friction regularizer: }r_F. \tag{原文 21}
$$

然后定义

$$
r_j\equiv\frac{r_F\mu_j^{-2}}{\langle\mu_j^{-2}\rangle},
\qquad
\hat\mu^2\equiv\frac{r_F}{r_0\langle\mu_j^{-2}\rangle}. \tag{原文 22}
$$

这里 $\langle\cdot\rangle$ 表示对摩擦维度 $j$ 求平均。这样既保证 $\mu_j^2r_j=\hat\mu^2r_0$，又保证 $r_F=\langle r_j\rangle$。Algorithm 5 可以概括为：先检查 $y$ 是否已在摩擦锥内；若摩擦方向全零则返回原点；否则用 Eq. (22) 计算 $r_j$ 与 $\hat\mu^2$，用 Eq. (16) 得到 $\hat y$，再分别处理 $\hat y_0=0$ 的特殊式 (20) 或一般式 (18)-(19)，最后反变换 $x_k=\hat x_k/\sqrt{r_k}$。

### 13.7 冲量动力学分析与调参：原文 Eq. (23)-(28)

当所有不等式约束在解处不活跃，即 $\phi(f)>0$，物理上近似非滑动接触时，原文由 Eq. (15) 得到

$$
f=R^{-1}(v^*-Jv'). \tag{原文 23}
$$

这个式子与完整约束冲量 Eq. (13) 很像，所以原文把完整约束和接触堆叠到一个 combined space 中。令

$$
\mathcal J\equiv
\begin{bmatrix}
J_E\\ J
\end{bmatrix},
\qquad
\mathcal R\equiv
\begin{bmatrix}
R_E&0\\0&R
\end{bmatrix},
\qquad
\dot x^*\equiv
\begin{bmatrix}
v_E^*\\v^*
\end{bmatrix}.
$$

为了避免和原文重复使用 $J,R$ 造成混淆，这里用 $\mathcal J,\mathcal R$ 表示堆叠后的 Jacobian 和正则矩阵。定义

$$
\dot x=\mathcal Jv,
\qquad
A=\mathcal JM^{-1}\mathcal J^T,
\qquad
 a=\mathcal JM^{-1}(p-c+u).
$$

期望下一步速度由二阶稳定器给出：

$$
\dot x^*\equiv \dot x-hB\dot x-hKx. \tag{原文 24}
$$

由此得到 combined space 动力学

$$
(I+A\mathcal R^{-1})\ddot x+A\mathcal R^{-1}B\dot x+A\mathcal R^{-1}Kx=a. \tag{原文 25}
$$

原文随后做一个有解释力的理想化分析：若令 $\mathcal R=A\epsilon$，并把阻尼和刚度写成 $B=B_0(1+\epsilon)$、$K=K_0(1+\epsilon)$，且暂时假设 $A$ 可逆，则上式化为

$$
\ddot x+B_0\dot x+K_0x=\frac{\epsilon}{1+\epsilon}a. \tag{原文 26}
$$

这说明 $\epsilon\to0$ 对应接近硬约束，约束误差主要由自主二阶系统消除；$\epsilon\to\infty$ 则更像被外部加速度驱动的 spring-damper。实际实现中不能令 $\mathcal R=A\epsilon$，因为 $A$ 可能奇异，而且为了保持解析反演，$R$ 需要对角。因此原文采用对角近似：

$$
R_{ii}=\max(A_{ii}\epsilon,r_{\min}). \tag{原文 27}
$$

阻尼与刚度取为

$$
B_{ii}=2(1+\epsilon)\kappa^{-1},
\qquad
K_{ii}=(1+\epsilon)\kappa^{-2}. \tag{原文 28}
$$

若进一步用 $A$ 的对角近似分析，就得到原笔记已经写过的近似临界阻尼形式：

$$
\ddot x+2\kappa^{-1}\dot x+\kappa^{-2}x
\approx
\frac{\epsilon}{1+\epsilon}a.
$$

对静止在地面、受重力 $g$ 的自由物体，稳态穿透近似为

$$
x\approx\frac{\kappa^2\epsilon g}{1+\epsilon}.
$$

原文最后还给出一个更接近精确临界阻尼的替代方案：先按 Eq. (27) 定义对角 $R$，再解矩阵方程

$$
A R^{-1}B = 2\kappa^{-1}(I+AR^{-1}),
$$

以及

$$
A R^{-1}K = \kappa^{-2}(I+AR^{-1}).
$$

如果 $A$ 可逆，可以精确求解；如果 $A$ 不可逆，则可用伪逆。这样得到的 $B,K$ 不再是对角矩阵，但更接近真正的临界阻尼。

### 13.8 最终缺漏判断

严格按原文公式编号看，当前笔记在补入本节后已经覆盖 Eq. (1)-(28) 的全部数学主线。仍需注意两个小差异：第一，笔记为了可读性把原文 $v^0$ 统一写成 $v'$，把原文的 $\tilde M,\tilde v$ 写成 $\widehat M,\hat v$，这不是内容缺漏；第二，原文没有 Appendix，所以不存在附录公式遗漏。如果以后你下载到另一个带 appendix、supplement 或 tech report 的 MuJoCo 版本，需要另行核对，因为当前本地 PDF 并不包含这些内容。

## 14. 按主题整理原文引用文献

> [!note]
> 这一节不是按原文 References 的编号顺序复述，而是按论文中的知识来源和问题链条重新分组。方括号编号仍保留原文编号，便于回到正文查找作者在何处引用。

### 14.1 速度步进与互补接触动力学：MuJoCo 要摆脱的主流基线

[1] D. Stewart and J. Trinkle, “An implicit time-stepping scheme for rigid-body dynamics with inelastic collisions and coulomb friction,” *International Journal for Numerical Methods in Engineering*, vol. 39, pp. 2673–2691, 1996. 这是刚体接触仿真中 velocity-stepping / time-stepping 思路的经典来源之一。它把碰撞和库仑摩擦接触放进隐式时间步进框架中处理，避免显式事件检测带来的不稳定，是 MuJoCo 论文开头所说“velocity-stepping complementarity approach”的代表性背景。

[2] M. Anitescu, F. Potra, and D. Stewart, “Time-stepping for three-dimensional rigid body dynamics,” *Computer Methods in Applied Mechanics and Engineering*, vol. 177, pp. 183–197, 1999. 这篇与 [1] 同属三维刚体接触的时间步进方法谱系，强调用离散时间中的互补问题处理接触和摩擦。MuJoCo 论文引用它，是为了说明在 MuJoCo 之前，主流稳定仿真路线已经从朴素 spring-damper 转向互补式 time-stepping。

[3] D. Kaufman, S. Sueda, D. James, and D. Pai, “Staggered projections for frictional contact in multibody systems,” *ACM Transactions on Graphics*, vol. 164, pp. 1–11, 2008. 这篇代表图形学/多体系统中处理摩擦接触的 staggered projection 方法。MuJoCo 论文一方面用它说明精确摩擦互补问题的计算困难，另一方面在 Introduction 中指出 Drumwright 与 Shell 的两步优化方法在结构上有点类似 staggered projection，因此不适合 MuJoCo 所追求的解析反演。

### 14.2 Complementarity-free 与接触模型验证：MuJoCo 的直接理论前身

[4] E. Todorov, “A convex, smooth and invertible contact model for trajectory optimization,” *ICRA*, 2011. 这是本文最直接的前作。当前 MuJoCo 论文把 [4] 的 convex、smooth、invertible contact model 扩展到完整仿真管线中，加入完整约束、干关节摩擦、限位、摩擦接触以及解析 inverse dynamics。可以把 [4] 看作“接触模型核心想法”，本文则是“完整 MuJoCo 实现与理论闭环”。

[5] E. Drumwright and E. Shell, “Modeling contact friction and joint friction in dynamic robotic simulation using the principle of maximum dissipation,” *International Workshop on the Algorithmic Foundations of Robotics*, 2010. 这篇也是 complementarity-free 接触建模方向的重要工作。MuJoCo 论文承认它与 [4] 独立发展出类似动机：放松严格互补，改用优化式接触模型；但作者也指出 [5] 使用两步优化，因此不容易像 MuJoCo 这样做解析反演。

[6] E. Drumwright and E. Shell, “An evaluation of methods for modeling contact in multibody simulation,” *ICRA*, 2011. 这篇用于支持一个实践判断：虽然放松严格互补会改变理想刚体模型，但在复杂机器人仿真中，近似接触模型未必更差，甚至可能更符合现象学接触的可用性。MuJoCo 论文引用它，是为了说明 complementarity-free 近似在合成测试和模型控制经验中表现可接受。

[8] A. Chatterjee and A. Ruina, “A new algebraic rigid body collision law based on impulse space considerations,” *Journal of Applied Mechanics*, 1998. 这篇从冲量空间角度讨论刚体碰撞律。MuJoCo 论文引用它的关键作用是削弱“互补模型就是金标准”的直觉：刚体接触/碰撞模型本来就是现象学模型，不同冲量空间规则都可能合理，最终需要实验接触数据验证。

### 14.3 多体刚体动力学算法：MuJoCo Phase I 的计算基础

[9] R. Featherstone, *Rigid Body Dynamics Algorithms*. Springer, 2008. 这是多体刚体动力学算法的标准教材，涵盖 CRB、RNE、Articulated Body Algorithm 等。MuJoCo 论文中的 Phase I，即计算 $M,D,c,p,J_E,J$，明显建立在这一算法传统上。作者特别说明 MuJoCo 没有直接使用 Featherstone 的 $O(n)$ forward dynamics，因为接触冲量阶段需要显式惯性矩阵及其分解；一旦已经有了这些量，RNE 反而更合适。

[10] R. Featherstone, “Efficient factorization of the joint-space inertia matrix for branched kinematic trees,” *International Journal of Robotics Research*, 2005. 这篇更具体地支撑论文中关于 branch-induced sparsity 的说法。MuJoCo 虽然在最坏情况下有 $O(n^3)$ 的惯性矩阵分解复杂度，但机器人运动链和树结构带来的稀疏性使实际性能接近线性算法。它解释了为什么 MuJoCo 可以在保留显式矩阵形式的同时仍然很快。

### 14.4 约束动力学与稳定化：软 Gauss 原理和参数调节的来源

[11] F. Udwadia and R. Kalaba, “A new perspective on constrained motion,” *Proceedings of the Royal Society*, 1992. 这是论文中 Gauss principle 相关推导的理论来源。MuJoCo 的完整约束处理不是直接从拉格朗日乘子形式出发，而是把约束加速度视为质量度量下的投影；然后再把硬投影软化成带 $R_E^{-1}$ 惩罚项的凸优化问题。理解 [11] 有助于理解为什么 Eq. (4)-(7) 是“最小修改无约束运动”的形式。

[12] J. Baumgarte, “Stabilization of constraints and integrals of motion in dynamical systems,” *Computer Methods in Applied Mechanics and Engineering*, 1972. 这是 Baumgarte stabilization 的经典来源。MuJoCo 论文的 $v^*$、$v_E^*$ 和 Section V 中的 $\dot x^*=\dot x-hB\dot x-hKx$，本质上是在离散时间冲量框架里引入二阶误差恢复机制。引用 [12] 是为了说明 MuJoCo 的软约束不是任意弹簧，而是有约束误差稳定化传统的。

### 14.5 控制、估计与优化应用：为什么解析 inverse dynamics 重要

[7] Y. Tassa, T. Erez, and E. Todorov, “Synthesis and stabilization of complex behaviors through online trajectory optimization,” *IROS*, 2012. 这篇代表在线轨迹优化和模型预测控制方向。MuJoCo 论文引用它，是为了说明 complementarity-free 近似接触已经在 model-based control 中显示出可用性。本文强调解析 inverse dynamics 可能给 MPC 内层优化带来数量级加速，[7] 就是这一路线的直接应用背景。

[13] K. Lowrey, Y. Tassa, T. Erez, S. Kolev, and E. Todorov, “Physically-consistent sensor fusion for contact-rich behaviors,” manuscript under review, 2014. 这篇用于说明解析 inverse dynamics 在状态估计中的用途。接触丰富行为的传感器融合不能只拟合观测，还要满足物理一致性；MuJoCo 的快速 inverse dynamics 可以作为先验或约束，判断一段观测运动是否需要合理的控制力与接触冲量。

[14] I. Mordatch, E. Todorov, and Z. Popovic, “Discovery of complex behaviors through contact-invariant optimization,” *SIGGRAPH*, 2012. 这篇是接触不变优化生成复杂全身动作的代表。MuJoCo 论文结尾提到，如果能把 [14] 这类离线复杂动作合成推进到 MPC 模式，可能会改变机器人控制和互动游戏。当前论文的解析 inverse dynamics 被作者视作补上这一数量级速度缺口的关键工具。

### 14.6 文献关系图：从接触仿真到控制优化

如果按问题链条看，这 14 篇文献可以整理成一条路线。第一层是传统刚体接触仿真的稳定数值基线：[1]、[2] 建立隐式时间步进互补框架，[3] 展示摩擦多体接触中的投影式求解。第二层是对严格互补的放松与替代：[4] 是 Todorov 自己的 convex / smooth / invertible 接触模型，[5] 是独立的 maximum dissipation 方向，[6] 评估不同接触模型，[8] 从冲量空间提醒我们刚体碰撞律并非唯一真理。第三层是多体动力学与约束处理的基础：[9]、[10] 提供 CRB/RNE/稀疏惯性分解的算法背景，[11] 提供 Gauss 约束运动视角，[12] 提供 Baumgarte 稳定化思想。第四层是 MuJoCo 真正关心的应用落点：[7] 指向在线轨迹优化与控制，[13] 指向物理一致状态估计，[14] 指向接触丰富复杂行为合成。

这组引用也反映了 MuJoCo 论文的定位：它不是单纯站在图形学仿真传统里追求视觉可信，也不是单纯站在刚体力学传统里追求严格互补，而是把多体动力学算法、凸接触近似、约束稳定化和最优控制需求合在一起，服务于“可快速反复调用、可反演、可优化”的物理引擎。
