---
title: "约束 Jacobian J 如何参与约束求解"
type: concept-note
tags: [robotics, dynamics, constraints, jacobian, mujoco, analytical-mechanics]
created: 2026-07-02
related: [[mojucu]]
---

# 约束 Jacobian $J$ 如何参与约束求解

> [!summary]
> 约束 Jacobian $J$ 的作用可以用一句话概括：它把广义坐标空间里的运动，投影到约束空间；它的转置 $J^T$ 则把约束空间里的力或冲量，映射回广义坐标空间。也就是说，$J$ 负责“看见约束被违反了多少、违反速度是多少”，$J^T$ 负责“把修正约束所需的力施加回系统”。


## 0. 参考来源与本文边界

这篇笔记不自己发明 MuJoCo 的约束 Jacobian 规则，而是按下面几个来源整理。第一，MuJoCo 官方文档 Computation 章节给出系统级记号：$J(q)$ 是 constraint Jacobian，保存在 `mjData.efc_J`；它把 joint velocity $v$ 映射到 constraint coordinates 中的速度 $Jv$，而 $J^T$ 把 constraint force $f$ 映射回 joint coordinates 中的力 $J^Tf$。第二，同一文档的 Constraint model 章节明确说：每个 active scalar constraint 对应 $J$ 的一行，active constraints 按 equality、friction loss、limit、contact 排列；equality 的 $J$ block 是 residual $r(q)$ 的 Jacobian，也就是 $\partial r/\partial q$。第三，MuJoCo 官方 Contact / Friction cones 小节说明 contact Jacobian 的构造：先构造 $6\times n_v$ 的空间 Jacobian 差分矩阵 $S$，它把 joint velocity 映射到 contact frame 下的相对空间速度，然后用接触力基矩阵 $E$ 得到 contact Jacobian $E^TS$。第四，Todorov 的 MuJoCo 论文使用同一套思想，把完整约束写成 $J_E$，接触/限位/摩擦写成 $J$，并在离散动力学中以 $J^Tf$ 和 $J_E^Tf_E$ 的形式进入方程。

> [!info] 主要参考
> - MuJoCo Documentation, **Computation / General framework**：$J$ 的系统级定义、`efc_J`、$Jv$ 与 $J^Tf$ 的含义。  
> - MuJoCo Documentation, **Computation / Constraint model**：active constraint 类型、equality/friction loss/limit/contact 的 residual 和 Jacobian 规则。  
> - MuJoCo Documentation, **Computation / Friction cones**：contact Jacobian 的 $S$、$E$、$E^TS$ 构造。  
> - Emanuel Todorov, **Convex and analytically-invertible dynamics with contacts and constraints: Theory and implementation in MuJoCo**：论文中的 $J_E,J,f_E,f,A=J\widehat M^{-1}J^T$ 记号和动力学推导。

## 0.1 各种约束情况下 $J$ 怎么定义

MuJoCo 的统一思路是：先为每个约束定义一个 residual 或约束空间速度，再对广义速度建立线性映射。系统级 $J$ 不是单个几何对象的 Jacobian，而是把所有 active scalar constraints 的行堆叠起来得到的矩阵。官方文档用 $n_c$ 表示 active constraints 数量，所以系统级

$$
J(q)\in\mathbb R^{n_c\times n_v}.
$$

每一行或每一小块都来自某一个约束对象。不同情况如下。

| 约束类型 | residual / 约束量怎么定义 | $J$ 怎么求 | 备注 |
|---|---|---|---|
| Equality: general | $r(q)=0$，$r$ 是可微标量或向量 residual | $J_r=\partial r/\partial q$，但由于 quaternion 等原因，MuJoCo 最终得到的是 $n_v$ 维速度空间中的 Jacobian 行 | 官方文档明确说 equality block 是 residual 的 Jacobian |
| Equality: connect | 两个 body 上 anchor 点的全局 3D 位置差 | 分别求两个 anchor 点对 $q$ 的 translational Jacobian，然后相减 | residual 维度为 3，相当于球铰外置闭链 |
| Equality: weld | 两个 body 的相对位姿误差，3D 位置误差 + 3D 姿态误差 | 位置部分用两个点/坐标系的 translational Jacobian 差；姿态部分用 relative angular velocity Jacobian | residual 维度为 6；官方文档说明姿态 residual 用 $\sin(\theta/2)(x,y,z)$ |
| Equality: joint | 标量关节位置之间的等式或多项式耦合 | 对关节位置 residual 对 $q$ 求导；若只锁一个 scalar joint，Jacobian 基本是选出该关节自由度的单位行 | 官方文档给出 quartic polynomial coupling 形式 |
| Equality: tendon | tendon length residual | tendon length 对 $q$ 的梯度，也就是 tendon moment arm vector | 官方文档说 tendon length 对位置的依赖可线性也可由 wrapping 计算 |
| Friction loss: scalar joint | 没有 position residual，形式上 $r(q)=0$；约束目标是抑制该 joint velocity | 对 scalar joint，$J$ 是一行选择向量：在该 joint address 上为 1，其余为 0 | 官方文档明确说 friction loss 没有位置 residual，但 affected joint/tendon velocity 作为 velocity residual |
| Friction loss: ball/free joint | 对 ball joint 的 3 个速度自由度或 free joint 的 6 个速度自由度分别施加 friction loss | $J$ 是选出对应 angular/linear velocity 子空间的多行选择块 | 每个自由度独立施加 friction loss |
| Friction loss: tendon | 无 position residual；抑制 tendon length velocity | $J=\partial l(q)/\partial q$，即 tendon moment arm vector | 与 tendon actuator 的 moment arm 是同类几何量 |
| Limit: joint | residual 是当前 joint position 到最近 limit value 的有符号距离；未到达为正，到达为 0，违反为负 | 对 residual 求导；scalar joint limit 通常是 $+1$ 或 $-1$ 的选择行，符号取决于上限/下限 | limit 是单边约束，active 后类似 frictionless contact |
| Limit: tendon | residual 是 tendon length 到 limit 的有符号距离 | $J=\pm\partial l(q)/\partial q$，符号取决于上限/下限 | 官方文档说 limits 和 contacts 都有 spatial residual，且是 unilateral |
| Contact: frictionless normal | residual 是两个 geom 表面距离；contact frame 第一轴为 normal | 先求 contact point 作为 geom1 与 geom2 上点的空间 Jacobian，相减得到 $S$；只取 normal force basis，$J=E^TS$ 的第一行 | `condim=1`，只产生法向 scalar constraint |
| Contact: regular friction | 约束空间包含 normal + 两个 tangent directions | 仍先构造 $S$，再用接触 frame 中的 basis matrix $E$ 投影：$J=E^TS$ | elliptic cone 中 $E$ 近似 identity；pyramidal cone 中 $E$ 是 pyramid edges basis |
| Contact: torsional friction | 在 normal + tangential 基础上增加绕 contact normal 的相对角速度约束 | $S$ 包含 3D force velocity 与 3D angular velocity；torsional row 从 angular part 中取 contact normal 方向 | `condim=4` |
| Contact: rolling friction | 再增加两个 rolling angular directions | $E^TS$ 中加入 contact frame 两个切向转动方向的 torque basis | `condim=6`，可抑制滚动 |

这里最容易混淆的是 contact Jacobian。它不是简单的 $\partial d/\partial q$ 一行就完事，除非法向无摩擦接触只取 normal distance。一般接触要处理法向、切向、扭转和滚动，所以 MuJoCo 先构造一个相对空间速度映射

$$
S(q)\in\mathbb R^{6\times n_v},
$$

使得

$$
S(q)v
$$

是在 contact frame 中表达的两个 geom 接触点的相对空间速度。然后给每个接触力/力矩分量定义一个 6D basis，列成矩阵 $E$。最终接触 Jacobian 是

$$
J_{contact}=E^TS.
$$

这条公式是 MuJoCo 官方文档给出的 contact Jacobian 构造。直观地说，$S$ 告诉你“两个接触点相对怎么动”，$E^T$ 告诉你“我们关心相对运动在法向、切向、扭转、滚动这些约束方向上的哪些分量”。

## 0.2 两个具体例子：不是抽象编造，而是按 residual 求导

第一个例子是 scalar hinge joint 的下限。假设某个 hinge joint 位置是 $q_i$，下限是 $q_{min}$。MuJoCo 文档说 limit residual 是当前位置到最近 limit value 的有符号距离，未到达为正，违反为负。对下限，可以取

$$
r(q)=q_i-q_{min}.
$$

当 $q_i>q_{min}$ 时 residual 为正；当 $q_i=q_{min}$ 时为零；当 $q_i<q_{min}$ 时为负。于是 active lower-limit constraint 的 Jacobian 行就是

$$
J=\frac{\partial r}{\partial q}
=\begin{bmatrix}0&\cdots&1&\cdots&0\end{bmatrix}.
$$

如果是上限 $q_{max}$，为了保持“未到达为正、违反为负”，可以取

$$
r(q)=q_{max}-q_i,
$$

于是 Jacobian 行变成

$$
J=\begin{bmatrix}0&\cdots&-1&\cdots&0\end{bmatrix}.
$$

第二个例子是两个 body 上两个 anchor 点的 connect equality。设两个 anchor 的世界坐标分别是 $x_A(q),x_B(q)\in\mathbb R^3$，residual 定义为

$$
r(q)=x_A(q)-x_B(q).
$$

那么

$$
J=\frac{\partial r}{\partial q}
=\frac{\partial x_A}{\partial q}-\frac{\partial x_B}{\partial q}
=J_A-J_B.
$$

这正是官方文档中 “connect residual 是两个全局 3D 点位置差” 的直接导数形式。它也和 contact 中的 $S$ 构造思路一致：先把两个点各自的空间 Jacobian 算出来，再相减得到相对运动的 Jacobian。

## 0.3 MuJoCo 里如果要拿 $J$，对应哪个数据字段

MuJoCo 官方文档给出的系统级字段是：

| 数学符号 | MuJoCo 字段 | 含义 |
|---|---|---|
| $J(q)$ | `mjData.efc_J` | active constraints 的系统级 constraint Jacobian |
| $r(q)$ | `mjData.efc_pos` | constraint residual |
| $f(q,v,\tau)$ | `mjData.efc_force` | constraint force |
| $n_c$ | `mjData.nefc` | active scalar constraints 数量 |

因此，如果你在 MuJoCo 运行时看 $J$，看的不是某个固定模型元素的 Jacobian，而是当前 active constraints 的 Jacobian。接触是否 active、limit 是否 active、margin/gap 设置，都会改变 `nefc` 和 `efc_J` 的行数与内容。

## 1. 为什么需要 Jacobian

机器人或刚体系统通常不用每个质点的笛卡尔坐标描述，而是用广义坐标 $q$ 描述。例如机械臂的 $q$ 是一串关节角，移动机器人可能有位置、朝向和轮子角度，带自由基座的人形机器人还会有基座位姿。可是约束往往不是直接写在关节角上的，而是写在某个几何量上，比如末端必须在桌面上，两个连杆端点必须重合，接触点不能穿透地面，轮子不能侧滑。

这就产生了一个桥接问题：约束在几何空间里，而动力学在广义坐标空间里。Jacobian $J$ 就是这座桥。它告诉我们，当广义坐标发生一个很小变化 $\delta q$ 时，约束量会发生多大变化。若约束函数写成

$$
\phi(q)=0,
$$

那么一阶线性化给出

$$
\delta \phi \approx \frac{\partial \phi}{\partial q}\delta q.
$$

我们把

$$
J(q)=\frac{\partial \phi}{\partial q}
$$

称为约束 Jacobian。因此，$J$ 的最基本含义是：它是约束函数对广义坐标的导数。

## 2. 从位置约束到速度约束

如果系统满足完整约束

$$
\phi(q,t)=0,
$$

那么这个等式不仅在当前位置要成立，在运动过程中也要一直成立。对时间求导，得到

$$
\frac{d}{dt}\phi(q,t)
=
\frac{\partial \phi}{\partial q}\dot q+
\frac{\partial \phi}{\partial t}=0.
$$

如果约束不显含时间，也就是 $\frac{\partial \phi}{\partial t}=0$，那么速度约束就是

$$
J(q)\dot q=0.
$$

在机器人动力学里常把广义速度写成 $v$，于是写成

$$
J(q)v=0.
$$

这句话的物理意义非常直接：虽然系统可以有很多广义速度方向，但真正允许的速度必须位于约束曲面的切空间内。$Jv$ 就是在测量“当前速度是否正在离开约束面”。如果 $Jv=0$，说明速度沿着约束面滑动，不破坏约束；如果 $Jv\ne0$，说明系统正在朝违反约束的方向运动。

> [!example] 圆周运动的例子
> 一个质点被限制在半径为 $R$ 的圆上，约束是
>
> $$
> \phi(x,y)=x^2+y^2-R^2=0.
> $$
>
> 此时
>
> $$
> J=\begin{bmatrix}2x&2y\end{bmatrix}.
> $$
>
> 速度约束 $Jv=0$ 就是
>
> $$
> 2x\dot x+2y\dot y=0.
> $$
>
> 这说明速度 $(\dot x,\dot y)$ 必须与半径方向 $(x,y)$ 垂直，也就是只能沿圆的切线方向运动。

## 3. 从速度约束到加速度约束

如果速度约束是

$$
J(q)v=0,
$$

再对时间求导，就得到加速度层面的约束。注意 $J(q)$ 本身也随 $q$ 变化，所以

$$
\frac{d}{dt}(J(q)v)=J(q)\dot v+\dot J(q,v)v=0.
$$

因此，加速度必须满足

$$
J(q)\dot v=-\dot J(q,v)v.
$$

更一般地，如果约束显含时间，或者我们希望加入稳定化项，就会写成

$$
J(q)\dot v=a^*.
$$

这里 $a^*$ 是约束空间中的目标加速度。它可能来自几何项 $-\dot Jv$，也可能加入 Baumgarte stabilization，用来把数值误差慢慢拉回约束面。在 MuJoCo 的论文中，完整约束部分写成类似

$$
J_Ea=a_E^*.
$$

其中 $a$ 是广义加速度，$J_E$ 是 equality constraint Jacobian，$a_E^*$ 是期望的约束空间加速度。

## 4. $J^T$ 为什么把约束力映射回广义力

前面说 $J$ 把广义速度映射到约束空间速度。现在反过来，如果约束空间里有一个力或冲量 $f$，它应该如何作用到广义坐标上？答案是通过 $J^T$。

这个结论来自虚功原理。假设广义虚位移是 $\delta q$，那么约束空间中的虚位移是

$$
\delta x=J\delta q.
$$

如果约束空间中的力是 $f$，它做的虚功是

$$
\delta W=f^T\delta x.
$$

把 $\delta x=J\delta q$ 代入，得到

$$
\delta W=f^TJ\delta q=(J^Tf)^T\delta q.
$$

而广义力 $\tau_c$ 的定义正是

$$
\delta W=\tau_c^T\delta q.
$$

所以必须有

$$
\tau_c=J^Tf.
$$

这就是为什么约束力、接触力、接触冲量总是以 $J^Tf$ 的形式进入广义坐标动力学。$J$ 负责从广义运动到约束运动，$J^T$ 负责从约束力到广义力。

> [!important] 记忆方式
> $Jv$ 是“这个广义速度在约束方向上有多快”；$J^Tf$ 是“这个约束方向上的力对每个广义坐标产生多少等效力矩”。

## 5. 用 $J$ 求约束力：拉格朗日乘子形式

考虑无接触的等式约束系统。动力学写成

$$
M\dot v=\tau+J^T\lambda,
$$

其中 $\lambda$ 是约束力的拉格朗日乘子。与此同时，约束要求

$$
J\dot v=a^*.
$$

我们要解的是 $\dot v$ 和 $\lambda$。把动力学改写为

$$
\dot v=M^{-1}(\tau+J^T\lambda).
$$

代入约束加速度条件，得到

$$
J M^{-1}(\tau+J^T\lambda)=a^*.
$$

展开后是

$$
JM^{-1}J^T\lambda=a^*-JM^{-1}\tau.
$$

如果 $JM^{-1}J^T$ 可逆，那么

$$
\lambda=(JM^{-1}J^T)^{-1}(a^*-JM^{-1}\tau).
$$

这个公式展示了约束求解的基本结构。首先，$M^{-1}\tau$ 给出无约束加速度；其次，$J$ 检查这个加速度在约束空间里违反了多少；最后，$J^T\lambda$ 施加约束力，把加速度修正到满足约束的位置。

## 6. 接触空间矩阵 $A=JM^{-1}J^T$ 是什么

上面出现的矩阵

$$
A=JM^{-1}J^T
$$

非常重要。在 MuJoCo 论文里，更准确地说，完整约束被软化吸收后，会出现

$$
A=J\widehat M^{-1}J^T.
$$

它叫做接触空间的 inverse apparent inertia。可以把它理解成：在约束空间施加单位冲量，会造成多少约束空间速度变化。

假设一个接触冲量是 $f$。它在广义坐标中产生的速度变化是

$$
\Delta v=M^{-1}J^Tf.
$$

把这个速度变化投影回接触空间，得到

$$
J\Delta v=JM^{-1}J^Tf=Af.
$$

所以 $Af$ 就是接触冲量 $f$ 导致的接触空间速度变化。若 $A$ 的某个方向很小，说明那个方向上系统表观惯性大，冲量很难改变速度；若 $A$ 的某个方向很大，说明系统在那个方向上比较“轻”。

## 7. MuJoCo 中 $J$ 的求解流程

MuJoCo 论文的离散动力学主方程是

$$
M(v'-v)=(p+u-c)h+J^Tf+J_E^Tf_E.
$$

这里 $J_E$ 是等式约束的 Jacobian，$J$ 是接触、限位、摩擦等约束的 Jacobian。MuJoCo 的处理流程可以分成两步。第一步，先用软 Gauss 原理处理完整约束，把 $J_E$ 的影响吸收到修正后的表观惯性 $\widehat M$ 和预测速度 $\hat v$ 里：

$$
\widehat M=M+J_E^TR_E^{-1}J_E.
$$

然后得到

$$
v'=\hat v+\widehat M^{-1}J^Tf.
$$

第二步，把这个式子左乘 $J$，投影到接触空间：

$$
Jv'=J\hat v+J\widehat M^{-1}J^Tf.
$$

定义

$$
v^+=Jv',\qquad v^-=J\hat v,\qquad A=J\widehat M^{-1}J^T,
$$

就得到 MuJoCo 的接触空间动力学：

$$
Af+v^-=v^+.
$$

这个式子说明，接触求解已经从广义坐标空间里的大动力学问题，变成了接触空间里的冲量问题。接下来只需要给 $f$ 加上法向非负、摩擦锥、限位等约束，并通过凸优化求解。

## 8. 为什么 $J$ 可以同时处理接触、限位和摩擦

关键在于 $J$ 不是只表示一种约束，而是表示“某个约束方向上的速度如何由广义速度产生”。对于关节限位，$J$ 可能只是选出某个关节速度；对于接触法向，$J$ 表示接触点相对速度在法向上的分量；对于滑动摩擦，$J$ 表示接触点相对速度在切向上的分量；对于扭转摩擦或滚动摩擦，$J$ 表示相对角速度在对应方向上的分量。

因此，不同物理约束可以统一写成

$$
v_{constraint}=Jv.
$$

它们的冲量也统一写成

$$
\tau_{constraint}=J^Tf.
$$

不同之处只在于 $f$ 的可行集合不同：法向接触要求 $f_n\ge0$，干摩擦要求 $-\eta\le f_i\le\eta$，摩擦接触要求冲量落在摩擦锥中。MuJoCo 正是利用这一点，把多种约束统一到同一个 impulse-space convex optimization 里。

## 9. 一个简单例子：质点被地面约束

设一个二维质点的坐标是

$$
q=\begin{bmatrix}x\\y\end{bmatrix},
$$

地面是 $y\ge0$。当质点接触地面时，法向约束方向就是 $y$ 方向。接触法向速度为

$$
v_n=\dot y.
$$

因此 Jacobian 是

$$
J=\begin{bmatrix}0&1\end{bmatrix}.
$$

若法向冲量是 $f_n$，那么作用回广义坐标的冲量为

$$
J^Tf_n=\begin{bmatrix}0\\1\end{bmatrix}f_n.
$$

这说明地面对质点只施加竖直向上的冲量，不施加水平冲量。如果再加入切向摩擦，那么切向 Jacobian 是

$$
J_t=\begin{bmatrix}1&0\end{bmatrix},
$$

摩擦冲量 $f_t$ 通过 $J_t^Tf_t$ 作用回水平坐标。把法向和切向叠起来，就是

$$
J=\begin{bmatrix}0&1\\1&0\end{bmatrix},
\qquad
f=\begin{bmatrix}f_n\\f_t\end{bmatrix}.
$$

于是总接触冲量为

$$
J^Tf=\begin{bmatrix}f_t\\f_n\end{bmatrix}.
$$

这个小例子展示了 $J$ 的本质：它不是神秘矩阵，只是在问“哪个广义速度分量对应约束方向的速度”。

## 10. 最终图景

可以把约束 Jacobian 的角色总结成一条链：

$$
\text{几何约束 }\phi(q)=0
\quad\xrightarrow{\partial/\partial q}\quad
J(q)
\quad\xrightarrow{\times v}\quad
\text{约束速度 }Jv
\quad\xrightarrow{\text{求解冲量 }f}\quad
J^Tf
\quad\xrightarrow{}\quad
\text{修正广义动力学}.
$$

在 MuJoCo 这篇论文里，$J$ 的作用尤其核心。它先把完整约束通过 $J_E$ 吸收到 $\widehat M$ 和 $\hat v$ 中，再把接触、限位和摩擦通过 $J$ 投影到接触空间，形成

$$
Af+v^-=v^+.
$$

然后 MuJoCo 不再用严格互补条件决定 $f$，而是把 $f$ 放进一个带正则项的凸优化问题中求解。换句话说，$J$ 负责把几何约束翻译成代数动力学问题，而凸优化负责在这些约束下选择一个稳定、唯一、可反演的冲量。
