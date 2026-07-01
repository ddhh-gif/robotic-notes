---
title: ArtHOI — 单目视频4D手-关节物体交互重建
tags:
  - robotic-manipulation
created: 2026-06-25
type: permanent
summary: 用多个基础模型协同优化，从单目RGB视频重建手与关节物体的4D交互，无需预扫描模板。
---

**来源** arXiv:2603.25791v1，2026-03-26，Harbin Institute of Technology
**作者** Zikai Wang, Zhilu Zhang（通讯）, Yiqing Wang, Hui Li, Wangmeng Zuo
**项目页** https://arthoi-reconstruction.github.io


你用手机拍了一段视频，视频里你用手操作剪刀。ArtHOI 看完这段视频后能告诉你：①剪刀在空间里的真实大小和位置，②两个刀片各自怎么运动，③哪根手指碰到了哪里。只需要普通视频，不需要提前扫描剪刀。

**optimization-based framework that reconstructs 4D hand-articulated-object interactions from monocular videos via integrating and refining priors from multiple foundation models.**
组合现有基础模型 实现 单目摄像头手-物体分割


**核心问题**

已有 HOI 重建方法：要么只处理刚体，要么要求预扫描物体（多视角环绕拍摄）。关节物体（scissor、stapler、CD drive）从单目视频重建 4D 交互 = 未解决问题。
难在：没有真实尺度、深度模糊、手遮挡物体、物体各部件互相遮挡、没有物体模板。

**方法：四阶段 pipeline**

**Stage 1 数据预处理**

从单目输入视频 $\mathcal{V} = \{\mathbf{I}_i\}_{i=1}^N$$N$帧 RGB
用基础模型提取先验：
- 物体 mask：Segment-Anything 2 [54] 得到$\{M_i\}^{N}_{i=1}$ 
- human mask\[54\]: SAM
- 手部 mesh：WiLoR [52]（MANO 参数化）
- 深度图 $\mathbf{D}_i$：Video-Depth-Anything [6] 
- 相机内参 $\mathbf{K}$：UniDepthV2 [51]
- 去人体遮挡：DiffuEraser inpainting [30] → 得到只含物体的 inpainted 视频 $\mathcal{V} = \{I^′_i\}^N_{i=1}$ 
-  inpainted video 经相同pipeline extract object-only masks  and depth maps 
HunYuan3D [26] 在 inpainted canonical frame 上生成 normalized mesh $\mathcal{G}^o$（无真实尺度）。

**Inpainting** is a digital editing process where specific areas of an image are seamlessly filled in, modified, or removed. It relies on a "mask" to define the target region, allowing you to easily erase unwanted objects, repair damaged photos, or generate new elements that perfectly match the surrounding context

整理如下(符号沿用论文记法):

| 变量                                     | 含义                              | 产生来源                           |
| -------------------------------------- | ------------------------------- | ------------------------------ |
| $\mathcal{V}={\mathbf{I}_i}_{i=1}^N$   | 输入单目视频,$N$ 帧 RGB 图像             | 原始输入                           |
| ${\mathbf{M}_i}_{i=1}^N$               | 物体掩码                            | 视频分割模型 [54] Segment-Anything 2 |
| (人体掩码)                                 | 人手/人体掩码                         | 同一视频分割模型 [54]                  |
| ${\mathbf{D}_i}_{i=1}^N$               | 度量深度图(metric depth)             | 单目深度估计器   Video-Depth-Anything |
| $\mathbf{K}$                           | 相机内参                            | 单目深度估计器  给出 UniDepthV2         |
| $\mathcal{V}'={\mathbf{I}'_i}_{i=1}^N$ | 去人后的 inpainted 视频(仅含物体)         | 视频 inpainting 模型 [30] 移除人手遮挡   |
| ${\mathbf{M}'_i}_{i=1}^N$              | 仅物体的掩码                          | 对 $\mathcal{V}'$ 重跑同一预处理管线     |
| ${\mathbf{D}'_i}_{i=1}^N$              | 仅物体的深度图                         | 同上(对 $\mathcal{V}'$ 重跑)        |
| $\mathbf{I}'_c$                        | inpainted 的标准帧(canonical frame) | 从 $\mathcal{V}'$ 选定/指定的规范视角帧   |
| $\mathbf{M}'^o_c$                      | $\mathbf{I}'_c$ 上的物体掩码          | 由标准帧对应的物体掩码裁出物体图               |
| (物体 3D mesh)                           | 关节物体的完整几何                       | 把裁剪后的物体图喂给 HunYuan3D [26]      |

**3.2 Metric Pose and Scale Optimization**

|变量|含义|产生来源|
|---|---|---|
|(normalized mesh)|HunYuan3D 输出的归一化网格|3.1 的 HunYuan3D 重建|
|$\mathbf{D}'^o_c$|标准帧的物体度量深度|3.1 预处理(物体深度)|
|$\mathbf{M}'^o_c$|标准帧的物体掩码|3.1 预处理|
|$s^o_c$|物体的度量尺度(metric scale)|优化变量(待求)|
|$\mathbf{T}^o_c$|物体的 6-DoF 位姿|优化变量(待求)|
|(metric canonical mesh)|世界坐标下的度量规范网格|用 $s^o_c,\mathbf{T}^o_c$ 对齐归一化网格得到|
|ASR|Adaptive Sampling Refinement,自适应采样细化|本文方法,用反投影度量深度先算粗尺度,再在自适应范围内迭代采样候选尺度|

几点便于你索引的提示:
上标 $o$ = object,下标 $c$ = canonical,撇号 $'$ = inpainted(去人后)管线产物。这套命名是"${\text{量}}^{\text{对象}}_{\text{帧}}$ + 是否 inpainted"的正交组合,认准这三个维度就能快速还原任意符号含义。

FoundationPose [61] 在文中是被否定的 baseline(直接用它做 6-DoF 估计,因 mesh 与深度不一致而退化),不是本文采用的产生来源,所以没列进变量表——它对应的是 ASR 要解决的问题signature:"异质先验(生成的 mesh + 不准的深度)之间不一致 → 直接位姿估计不稳"。
![[Pasted image 20260629191533.png]]
 3.3 节 **Part-wise Motion Reconstruction(分部件运动重建)**。
**目标设定**
要同时利用**空间线索**(物体长什么样)和**时间线索**(它怎么动),还要处理**部件级遮挡**(运动中某部件被挡住)。做法:借助 dense tracking(密集点跟踪)先拿到粗糙的部件运动,再为每个部件优化随时间变化的 **SE(3) 变换**(刚体运动 = 旋转+平移,SE(3) 是它的数学群)。

**具体流程**

把第 $i$ 帧的部件掩码记作 ${\mathbf{M}'^{p_k}_i}_{k=1}^K$($K$ 个部件)。第一步用 **PartField [36]** 把标准物体 mesh $\mathcal{G}^o_c$ 切成若干部件——它给 mesh 顶点分组,这些分组对应到图像上就是各部件的掩码。

然后在 inpainted 视频 $\mathcal{V}'$ 上跑 **CoTracker [23]**,产生**时间连贯的点轨迹**和**每点可见性**。对第 $k$ 个部件:在它的掩码 $\mathbf{M}'^{p_k}$ 内采样 $Q$ 个查询像素,用 **CoTrackerV3** 跟踪这些点,得到 **2D 轨迹** + **逐帧可见性指示**。再用深度图 $\mathbf{D}'_i$ 把 2D 点**升维到 3D**,得到 3D 轨迹和可见性对 $(\mathbf{z}^k_{i,q},, v^k_{i,q})$,其中 $v^k_{i,q}\in{0,1}$(1=可见,0=被遮挡)。最后用一个轻量后处理**剔除离群轨迹**(跟错的点)。

**优化目标:两个损失**

要求解的是每帧每部件的 SE(3) 变换 ${\mathbf{T}^k_i}_{i=1}^N$,通过两个约束来定:

_轨迹一致性损失(式 1)_:

$$\mathcal{L}_{\text{track}}=\sum_{j\in\mathbb{S}}\sum_{q\in\mathbb{W}^k_{i,j}}\left|\mathbf{z}^k_{j,q}-(\mathbf{T}^{p_k}_i)^{-1}\mathbf{T}^{p_k}_j,\mathbf{z}^k_{i,q}\right|$$

直觉是:同一个物理点,在第 $i$ 帧的 3D 位置 $\mathbf{z}^k_{i,q}$,如果我用估出的两帧间相对变换 $(\mathbf{T}^{p_k}_i)^{-1}\mathbf{T}^{p_k}_j$ 把它从 $i$ 帧"搬"到 $j$ 帧,搬过去的位置应该和**实测的** $j$ 帧位置 $\mathbf{z}^k_{j,q}$ 重合。两者的差就是误差,最小化它就是在逼迫估出的 SE(3) 序列和真实点轨迹一致。

求和范围里 $\mathbb{S}$ 是采样的参考帧集合;$\mathbb{W}^k_{i,j}={q\mid v^k_{i,q}=1\wedge v^k_{j,q}=1}$ 是**在 $i$ 和 $j$ 两帧都可见**的点集合——这就是"可见性约束":只用两帧都看得见的点算损失,避开遮挡导致的错误监督。

_平滑损失(式 2)_:

$$\mathcal{L}_{\text{smooth}}=\sum_{i=2}^{N-1}\left|\Delta^2\mathbf{T}^{p_k}_i\right|$$

$\Delta^2$ 是**时间维上的二阶差分**算子:$\Delta^2\mathbf{T}^{p_k}_i=\mathbf{T}^{p_k}_{i+1}-2\mathbf{T}^{p_k}_i+\mathbf{T}^{p_k}_{i-1}$。二阶差分衡量的是"运动的加速度/抖动",最小化它等于**惩罚突变**,让部件运动随时间平滑,不要逐帧乱跳。这是对噪声跟踪结果的正则化。

$L_{motion} = L_{track} + λ_{smooth}L_{smooth}$. 
---

**两个值得记住的设计点**

第一,**可见性门控**($\mathbb{W}$ 的定义)是处理遮挡的关键技巧:不修复遮挡、不猜被挡的点,而是直接在损失里把不可见的点**排除**,只用可信的可见点做监督。这是个通用的"缺数据就别用,而不是硬补"的思路。

第二,这里的结构和你刚看的 ASR 是**同一类问题的两种解法**对照:ASR 是**采样搜索**(不可微分,撒点+打分),式 1/式 2 是**可微优化**(定义损失,梯度下降)。区别在于 ASR 要解的尺度只是一个标量、且裁判(IoU)不可导,所以用搜索;这里要解的是 $N$ 帧 ×$K$ 部件的连续 SE(3) 变量、损失可导,所以用优化。**这正好是"离散搜索 vs 连续优化"的选择signature**——变量维度高、损失可微 → 优化;变量低维、判据不可微 → 搜索。

需要的话,我可以把式 1 那个 $(\mathbf{T}_i)^{-1}\mathbf{T}_j$ 的几何含义(相对位姿、为什么要做逆)再展开画一下,或者把这两节按你的组件卡片框架整理成可检索条目。
**Stage 2 Adaptive Sampling Refinement (ASR) image-to-3D model to generate a normalized 3D mesh from the inpainted object.  **

解决 normalized mesh 没有 metric scale 和 pose 的问题。
直接用 FoundationPose [61] 估计 pose 时因为 mesh 和深度不一致会失败。ASR 的做法：

1. 粗尺度估计 $s^o_{\text{coarse}}$：把深度图反投影点云，取与 mask 对应点云在 x/y 轴方向的最大范围比值估算出物体尺度
2. 迭代采样（共 $J$ 次，代码里 $J=20$，初始 $\delta=0.03$）：
   - 随机采样 $s^o_c \leftarrow s^o_{\text{coarse}} \cdot (1 + \text{random} \in [-\delta, \delta])$
   - 按此 scale 缩放 mesh，用 FoundationPose 估计 6-DoF pose $\hat{\mathbf{T}}^o_c$
   - 渲染 silhouette 与 object mask 计算 IoU $\mathcal{L}_{\text{iou}}$
   - 若 IoU 提升则更新最优；若连续无提升则加倍 $\delta$
1. 输出最优 $s^o_c, \mathbf{T}^o_c$，Scale canonical mesh → $\mathcal{G}^o_c$

**ASR不直接信任单目深度算出的尺度,而是把"尺度"当成一个搜索问题——不停试不同的缩放值,谁让 mesh 投影出来的轮廓和真实 mask 最吻合,就选谁。**
Algorithm 1: Adaptive Sampling Refinement (ASR)
Require: Normalized object mesh $\mathcal{G}^o$; RGB $\mathbf{I}'_c$, depth $\mathbf{D}'_c$ and mask $\mathbf{M}'_c$ of canonical frame; camera intrinsics $\mathbf{K}$; number of iterations $J$; initial sampling range $\delta$
Ensure: Metric scale $s_c^o$ and pose $\mathbf{T}_c^o$ of canonical object, scaled canonical object mesh $\mathcal{G}_c^o$
1. $s_{\text{coarse}}^o \leftarrow \text{CoarseScaleEstimation}(\mathcal{G}^o, \mathbf{D}'_c, \mathbf{M}'_c)$
2. $(\mathcal{L}{\text{best}}, j{\text{best}}) \leftarrow (-\infty, 0)$
3. for $j = 1$ to $J$ do
    1. if $j_{\text{best}} < \dfrac{J}{2}$ then
        1. $\delta \leftarrow 2\delta$   // Adaptively expand the range
    2. end if
    3. $\hat{s}c^o \leftarrow s{\text{coarse}}^o \cdot \text{RandomSample}(-\delta, \delta)$
    4. $\hat{\mathcal{G}}_c^o \leftarrow \text{Scale}(\mathcal{G}^o, \hat{s}_c^o)$
    5. $\hat{\mathbf{T}}_c^o \leftarrow \text{FoundationPose}(\hat{\mathcal{G}}_c^o, \mathbf{I}'_c, \mathbf{D}'_c, \mathbf{M}'_c, \mathbf{K})$
    6. $\hat{\mathbf{M}}_c^o \leftarrow \text{RenderSilhouette}(\hat{\mathbf{T}}_c^o \cdot \hat{\mathcal{G}}_c^o, \mathbf{K})$
    7. $\mathcal{L}_{\text{iou}} \leftarrow \text{IoU}(\hat{\mathbf{M}}_c^o, \mathbf{M}'_c)$
    8. if $\mathcal{L}{\text{iou}} > \mathcal{L}{\text{best}}$ then
        1. $s_c^o \leftarrow \hat{s}c^o,\ \mathbf{T}c^o \leftarrow \hat{\mathbf{T}}c^o,\ \mathcal{L}{\text{best}} \leftarrow \mathcal{L}{\text{iou}},\ j{\text{best}} \leftarrow j$
    9. end if
4. end for
5. $\mathcal{G}_c^o \leftarrow \text{Scale}(\mathcal{G}^o, s_c^o)$
6. return $s_c^o, \mathbf{T}_c^o, \mathcal{G}_c^o$

**输入输出(Require / Ensure)**
输入:归一化 mesh $\mathcal{G}^o$、标准帧的 RGB $\mathbf{I}'_c$ / 深度 $\mathbf{D}'_c$ / 掩码 $\mathbf{M}'^o_c$、相机内参 $\mathbf{K}$、迭代次数 $J$、初始采样范围 $\delta$。 输出:度量尺度 $s^o_c$、位姿 $\mathbf{T}^o_c$、缩放好的 mesh $\mathcal{G}^o_c$。
第 1 行:反投影点云量跨度算粗尺度 $s^o_{\text{coarse}}$,作为搜索的中心。
第 2 行:初始化"历史最佳"。$\mathcal{L}_{\text{best}}$(最佳得分)设成 $-\infty$ , $j_{\text{best}}$(最佳出现在第几轮)设成 0。这是个"擂台"机制,后面谁更好谁上擂台。
第 3 行:主循环,迭代 $J$ 次。

第 4–6 行:**自适应扩大搜索范围**——这是 ASR 的"A(Adaptive)"。判断条件 $j_{\text{best}}<\frac{j}{2}$ 的意思是:如果到现在为止,最佳结果还停留在"前半程"(也就是最近这半程都没刷新过纪录),说明当前范围里好的尺度可能已经找完了、或者根本不在这个范围里 → 把采样范围 $\delta$ 翻倍($\delta\leftarrow2\delta$),去更大的范围里找。反过来,如果最近一直在刷新纪录,就不扩,继续在当前范围精挖。

第 7 行:**采样一个候选尺度**。在 $[-\delta,\delta]$ 里随机取一个值,乘到粗尺度上:$\hat{s}^o_c=s^o_{\text{coarse}}\cdot\text{RandomSample}(-\delta,\delta)$。也就是在粗尺度附近抖动出一个候选(注意这是带探索性的随机搜索,不是梯度下降)。

第 8 行:用这个候选尺度把归一化 mesh 缩放,得到候选 mesh $\hat{\mathcal{G}}^o_c$。

第 9 行:**对这个候选 mesh 跑位姿估计**。把缩放后的 mesh 喂给 FoundationPose,结合 RGB、深度、mask、内参,估出它的 6-DoF 位姿 $\hat{\mathbf{T}}^o_c$。

第 10 行:**渲染轮廓**。用估出的位姿把候选 mesh 摆好,投影到相机平面,渲染出它的剪影(silhouette)掩码 $\hat{\mathbf{M}}^o_c$——也就是"如果这个尺度+位姿是对的,物体在图里应该长这个形状"。

第 11 行:**算吻合度**。把渲染出的剪影 $\hat{\mathbf{M}}^o_c$ 和真实物体掩码 $\mathbf{M}'^o_c$ 求 IoU(交并比),作为这一轮的得分 $\mathcal{L}_{\text{iou}}$。IoU 越高,说明这个尺度+位姿越接近真相。

第 12–14 行:**擂台更新**。如果这一轮得分超过历史最佳,就把当前的尺度、位姿、得分、轮次都记下来,刷新纪录。

第 15 行:循环结束。

第 16 行:用最终胜出的尺度 $s^o_c$ 把原始归一化 mesh 缩放一次,得到最终度量 mesh $\mathcal{G}^o_c$。

第 17 行:返回尺度、位姿、mesh。

**为什么要这么绕(设计动机)**

直接路线("第 9 行 FoundationPose 单独跑一次就完事")在 3.2 已经说了会退化——因为生成的 mesh 和单目深度不一致,FoundationPose 在错误尺度下给出的位姿不可靠。ASR 的破解办法是:**把尺度的正确性外包给一个它信得过的、不依赖深度绝对值的判据——轮廓 IoU。** mesh 缩放对不对,最终体现在"投影出来的剪影和图像里物体的剪影对不对得上"。剪影匹配只看形状轮廓,绕开了深度尺度本身不准的问题,所以它能当裁判。

**两个值得记住的设计点(可迁移的问题signature)**

第一,这是**搜索 + 验证**结构:用一个廉价的几何裁判(IoU)去筛一个昂贵估计器(FoundationPose)的输出。当你"有一个不可靠的估计器 + 一个可靠但只能事后打分的验证器"时,就可以套这个模式——撒点、估计、打分、留最优。

第二,第 4–6 行的自适应扩张是个**自调节探索**技巧:用"最佳解是否长时间没更新"作为信号,来决定该扩大搜索还是该局部精挖。这比固定搜索范围更省、更鲁棒——范围太小可能根本没覆盖真值,范围太大又浪费采样;让它根据"最近有没有进步"自己长大,正好平衡探索与利用。















关键：ASR 100% success rate（Table 5），FoundationPose 直接用只有 60%，Any6D 也只有 60%。

  image-to-3D 模型（如 HunYuan3D）生成的 mesh 是 normalized
  的——放在一个标准化坐标系里，没有真实物理尺寸。但要把它和真实视频里的深度图、相
  机对上，必须知道它在世界空间里的真实 scale 和 6-DoF pose（位置+朝向）。

  直接拿 FoundationPose 估计 pose 为什么不行？因为 FoundationPose 预设输入 mesh
  已经有正确尺度，而 HunYuan3D 生成的 mesh
  和单目深度图之间存在系统性的尺度不一致，导致估计出来的 pose 不稳定，成功率只有
  60%。

  ASR 的做法
  把"找 scale"变成一个带自适应搜索范围的迭代采样-评估问题。
  第一步：粗估初始 scale
  把深度图反投影成 3D 点云，取物体 mask 对应的点云在 x/y 轴方向的最大范围，与
  normalized mesh 的对应尺寸做比值，得到粗估 $s^o_\text{coarse}$。

  第二步：迭代采样（共 $J=20$ 次，初始范围 $\delta=0.03$）

  每轮：
  1. 在 $[s^o_\text{coarse} \cdot (1-\delta),\ s^o_\text{coarse} \cdot(1+\delta)]$ 内随机采一个候选 scale $s^o_c$
  2. 按此 scale 缩放 mesh，喂给 FoundationPose 估计 6-DoF pose   $\hat{\mathbf{T}}^o_c$
  3. 用这个 pose 把 mesh 渲染成 silhouette(剪影)，与真实 object mask 计算 IoU $\mathcal{L}_\text{iou}$
  4. 若 IoU 比历史最优高 → 更新最优 $(s^o_c, \mathbf{T}^o_c)$；若连续无提升 → 把
  $\delta$ 加倍（扩大搜索范围）

  第三步：输出

  选 IoU 最高对应的 $(s^o_c, \mathbf{T}^o_c)$，将 canonical mesh 缩放到真实尺寸
  $\mathcal{G}^o_c$。


  为什么比直接用 FoundationPose 好
  FoundationPose 直接估计时，mesh
  和深度之间的尺度差异让渲染结果和真实图像严重不匹配，优化陷入错误局部极小。ASR
  用 silhouette IoU 作为外部反馈信号，把 scale 搜索和 pose
  估计解耦——先找到"渲染出来和真实轮廓最像的那个 scale"，pose
  估计只在正确尺度下运行，精度自然更高。

  结果（Table 5）：ASR 成功率 100%，FoundationPose 直接用只有 60%，Any6D 也只有
  60%。









**Stage 4 MLLM-guided HOI Alignment**

核心贡献二。解决手和物体空间不一致问题（分别重建时手-物体相互穿透或脱离）。

**三阶段 prompting**（Figure C/D/E）：
- Stage 1（prompt_perspective）：判断视频是第一人称还是第三人称，减少左右手混淆
- Stage 2a/2b（Hand Mapping）：第一人称用拇指方向判断左右手；第三人称用手臂连接方向
- Stage 3（Contact Reasoning）：逐帧分析，用 RGB+depth 彩色图联合判断接触状态和接触手指

拼接 $k=3$ 个相邻帧（RGB + 伪彩色深度图）为大图送入 MLLM（Qwen-VL-Max [2]），输出 JSON 包含每帧的 `r_contact`, `l_contact`, `r_fingers`, `l_fingers`。

**偏保守**：不确定时输出 `false`，因为 FP 比 FN 危害更大（FP 会给 mesh 加上错误的约束力）。

接触信息转化为优化约束：

$\mathcal{L}_{\text{contact}} = \sum_{i \in \mathbb{C}} \sum_{v_t \in \mathbb{T}_i} \min_{v_o \in \mathcal{G}^o_i} \left\| v_o - v_t \right\|_2$

其中 $\mathbb{T}_i$ 是 MLLM 判定接触帧的 MANO 指尖顶点集合。

加运动正则化 $\mathcal{L}_{\text{reg}}$ 约束手部参数不偏离初始估计（加速度先验 + $\ell_1$ 惩罚）。

两阶段优化：先固定手只优化物体 scale $s^o_c$（800 steps）；再联合优化 $s^o_c$、手姿态 $\theta^h$、全局变换 $\mathbf{T}^h$。

---

**数据集**

两个新数据集（本文贡献）：
- **ArtHOI-RGBD**：5 段视频，Intel RealSense，1280×720 @ 30FPS，有 GT depth
- **ArtHOI-Wild**：8 段网络视频，无 GT depth，更有挑战性

对比基准：RSRD [24]（9 个视频，需预扫描）、ARCTIC [11] 的 3 物体子集

---

**关键数字**

| 指标 | 含义 |
|------|------|
| CD (mm) ↓ | Chamfer Distance，物体重建精度 |
| MSSD (mm) ↓ | Maximum Symmetry-Aware Surface Distance |
| F5↑ / F10↑ | F-score at 5mm/10mm |
| $Co^2$ ↓ | Collision-Contact Score，HOI 对齐质量 |

Table 1 代表数据（ArtHOI-RGBD，Headphone）：

| 方法 | CD ↓ | MSSD ↓ | F10 ↑ |
|------|------|--------|-------|
| EasyHOI | 209.3 | 291.0 | 1.26 |
| RSRD | 14.7 | 41.1 | 41.67 |
| **ArtHOI** | **8.1** | **30.4** | **69.68** |

Table 4 HOI 对齐 $Co^2$（越低越好）：

| 方法 | ArtHOI-RGBD | ArtHOI-Wild |
|------|-------------|-------------|
| RSRD + WiLoR | 0.392 | N/A |
| Ours w/o MLLM | 0.046 | 0.059 |
| **Ours w/ MLLM** | **0.029** | **0.039** |

MLLM contact reasoning（ArtHOI-Wild，完整 prompting）：Acc=88.58%，FP=11.20%

运行时间：~1 小时/100 帧 @ 960×540，A6000 GPU。

---

**创新点**

1. **ASR** — 用迭代自适应采样解决 image-to-3D 模型输出没有真实尺度的固有问题，比直接用 pose estimator 鲁棒（100% vs 60% success rate）
2. **三阶段 MLLM prompting** — 视角检测 → 手映射 → 接触推理，每阶段消除一类幻觉来源；深度图彩色化提供近-远视觉线索
3. **联合 HOI 对齐优化** — contact loss + motion regularization 联合调整手和物体，解决分别重建后的空间错位

---

**费曼拷问：货物崇拜检测**

**有没有"演示>论证"的硬证据？**
有定量数字（Table 1-6）和视觉对比（Figure 3）。但 ArtHOI-RGBD 只有 5 段视频，ArtHOI-Wild 只有 8 段。样本量极小，结论的统计置信度存疑。RSRD 在 RSRD 自己的数据集上表现接近 ArtHOI（Table 2），表明 ArtHOI 优势主要在无法预扫描的场景下。

**哪些假设作者没自己验证？**
- HunYuan3D 生成 mesh 质量差时 pipeline 会怎样 → 没有 ablation
- SLERP 插值填充遮挡帧假设运动平滑 → 快速运动时会失效，未测试
- ArtHOI-Wild 的 $Co^2$ 评分缺乏 GT depth 支撑，如何获取标注不明
- pipeline 串联，各阶段误差累积效应未分析

**该说"不知道"的地方**
- 野外视频的接触标注可信度（没有 GT 深度，何来准确标注？）
- 1 小时运行时间离实时差三个数量级，作者用"can be accelerated"一句话带过，未提具体方案

---

**局限与待填坑**

- [ ] 数据集规模太小（5+8），需要更大规模验证
- [ ] 单次运行 ~1 小时，不适合实时或快速迭代
- [ ] 双手同时操作同一物体的处理细节不充分
- [ ] 物体快速运动/完全遮挡时鲁棒性未测试
- [ ] HunYuan3D mesh 质量对 ASR 的影响未量化
- [ ] 能否推广到非人手（如机械臂）？

---

**借用的经典研究**

- [[WiLoR]] — off-the-shelf 手部 MANO 参数估计 [52]
- FoundationPose [61] — 6-DoF pose estimation，ASR 内部调用
- HunYuan3D [26] — image-to-3D，生成 normalized canonical mesh
- CoTrackerV3 [23] — 稠密点追踪，提供 part motion 先验
- PartField [35/36] — per-vertex 特征场 + 聚类，物体部件分割
- Qwen-VL-Max [2] — MLLM，接触推理

**涌现现象**

三阶段 MLLM prompt 的累积效果（Table 6）：单独加 temporal 只改善 ~4%，再加 perspective + MinFP 大幅降低 FP（18.24 → 11.35），加 depth 进一步到 11.20。各模块叠加效果超过单独贡献之和，说明视角/深度/时序信息存在协同。




  深度（depth）：表示每个像素到相机的距离。
  通常是一张图，和 RGB 图一一对应，所以也叫深度图。
  点云（point cloud）：由很多 3D 点组成的数据集合。
  每个点通常有坐标 (x, y, z)，有时还带颜色、法向量等信息。
  网格（mesh）：在点云基础上，把相邻点连接成面，通常是三角形面片，形成连续表面。
  它比点云更像真正的“物体表面模型”。
  三者关系

  最常见的关系是：
  深度图 -> 点云 -> 网格
  1. 深度图 -> 点云
     把每个像素的深度值，结合相机内参，反投影到 3D 空间，就得到很多三维点。
  2. 点云 -> 网格
     再根据点之间的邻接关系，把这些点连成三角面，就得到 mesh。
  直观理解
  - 深度：告诉你“这个像素离相机多远”
  - 点云：把这些“远近信息”变成“空间中的很多点”
  - mesh：再把这些点拼起来，变成连续表面
  
  类比
  如果把物体表面看成一张纸：
  - 深度图：是在照片上记录纸上每个位置离你多远
  - 点云：是把纸表面撒很多采样点
  - mesh：是用线和三角形把这些点缝成完整的纸面
  -
  区别
  - 深度图是2.5D的，以相机视角存储，只能看到当前视角可见部分
  - 点云是真3D离散表示，但没有表面连接关系
  - mesh是真3D表面表示，有拓扑结构，更适合渲染、仿真、碰撞检测

  在机器人/视觉里常见流程
  RGB/视频 -> 估计深度 -> 转点云 -> 重建 mesh -> 用于抓取、定位、交互







# "分割 → 追踪 → 姿态优化" 流水线

## 一、这个套路的名字

**Test-time optimization on tracked correspondences**，或者更广义地叫 **analysis-by-synthesis via part-level SE(3) fitting**。

核心思想：不学姿态，而是在推理时直接用观测到的点轨迹作为约束，把变换参数 $T$ 当作优化变量求解。

---

## 二、流水线的三个固定阶段
### 阶段 1：空间分割（Segmentation）
给场景/物体打上结构性标签，把连续空间离散成若干语义区域。

|本题的选择|常见替代|
|---|---|
|PartField（neural field 级别的 part mask）|SAM / Mask2Former（2D mask）|
|canonical mesh 上的 K 个 part|点云聚类（K-Means, DBSCAN）|
||骨骼绑定（skinning weights）|

**不变的量**：输出是一组 mask ${M^{p_k}}$，索引后续所有操作的归属。
### 阶段 2：对应关系建立（Correspondence）

在时间轴（或视角轴）上，为每个 part 建立"哪个点在哪里"的匹配。

|本题的选择|常见替代|
|---|---|
|CoTrackerV3（2D pixel 追踪 → 深度图提升到 3D）|ORB/SIFT 特征匹配|
|query pixel 采样 + visibility mask $(z^k_{i,q}, v^k_{i,q})$|RAFT/FlowFormer 光流|
||DROID-SLAM 的 dense correlation volume|
||ICP 的最近邻对应|

**不变的量**：输出是一组带可见性标志的 3D 点对 ${(z_i, z_j, v)}$，作为下游优化的"观测"。

---

### 阶段 3：变换参数估计（Pose/Transform Fitting）

把对应关系转化为对 $T$ 的约束，加正则，求解。

**数据项**（观测一致性）： $$\mathcal{L}_{\text{data}} = \sum_{i}\sum_{j}\sum_{q \in \mathbb{W}_{i,j}} \bigl| z_{j,q} - T_i^{-1} T_j, z_{i,q} \bigr|$$

**正则项**（轨迹先验）： $$\mathcal{L}_{\text{reg}} = \sum_i \bigl| \Delta^n T_i \bigr|$$

|正则阶数 $n$|含义|典型场景|
|---|---|---|
|$n=1$（一阶差分）|惩罚速度突变|快速运动、短视频|
|$n=2$（二阶差分，本题）|惩罚加速度突变|较慢匀速运动|

**优化器**：

|方法|适用场景|典型配置|
|---|---|---|
|Adam + lr 线性衰减（本题）|参数量级差大（R vs t）；前端；可微渲染|500 iter，0.02→0.002|
|Gauss-Newton / LM|非线性最小二乘标准后端|10~50 iter 收敛|
|g2o / Ceres / GTSAM|大规模图优化；离线 BA|因子图结构|

---

## 三、这个套路在哪里反复出现

|方向|具体系统|分割|对应|优化|
|---|---|---|---|---|
|动态 SLAM|DROID-SLAM|—|dense flow|GRU + LM|
|动态场景重建|SUDS, D-NeRF|semantic mask|射线采样|Adam（per-scene）|
|人体姿态|SMPL fitting|body part|关键点|GN on joints|
|机器人操作|RGBManip, ManiGaussian|gripper mask|tracker|SE(3) fit|
|卫星 / 刚体追踪|本题 KBL 系列|part mask|CoTracker|Adam + smooth|
|NeRF 变形场|HyperNeRF, Nerfies|—|canonical warp|Adam|

---

## 四、套路的本质结构

```
场景/物体
    │
    ▼  [结构先验] ────────── 把空间离散成有意义的单元
K 个 part mask
    │
    ▼  [时序/多视角观测] ──── 建立跨帧/跨视角的点对应
N×Q 条轨迹 (z, v)
    │
    ▼  [优化] ──────────────── 以对应一致性为目标，轨迹平滑为约束
K×N 个 SE(3) 变换 T
```

三个阶段可以**完全独立替换**，接口只有两个：

- 阶段 1 → 阶段 2：mask $M^{p_k}$（决定在哪里采样）
- 阶段 2 → 阶段 3：轨迹 $(z^k_{i,q},, v^k_{i,q})$（决定残差怎么算）

---

## 五、设计时的关键决策点

**Q1：分割在 2D 还是 3D？** 2D mask（SAM）获取简单但深度提升引入噪声；3D/neural field（PartField）精度高但依赖 canonical reconstruction。

**Q2：对应是稀疏还是稠密？** 稀疏（特征点）计算快、受遮挡影响小；稠密（光流/CoTracker）信息量大但对遮挡敏感，需要 visibility mask $v$ 过滤。

**Q3：SE(3) 是 per-part 独立还是有铰链约束？** 独立（本题）：简单，适合刚体；带铰链（SMPL, URDF-based）：需要关节角参数化，适合有结构的关节体。

**Q4：优化是 test-time 还是 learned？** Test-time（本题）：泛化强，慢；Learned（端到端网络预测 $T$）：快，泛化受训练分布限制。

---

## 六、与你现有项目的对应

|你的项目|对应阶段|备注|
|---|---|---|
|RTAB-Map + D435i|阶段 2（特征追踪）+ 阶段 3（BA）|整个 SLAM 就是这条流水线的无分割版本|
|SCARA 末端追踪|阶段 1（link mask）+ 阶段 3（FK/IK fit）|关节约束替代平滑正则|
|KBL 卫星姿控|阶段 3（quaternion 优化）|无对应建立，直接用 IMU/星敏感器|