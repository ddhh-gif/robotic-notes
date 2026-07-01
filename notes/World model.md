 GAIA-1 (‘Generative AI for Autonomy’),
1 tokenlizer




**把自动驾驶的"世界模型"问题重新表述成 LLM 式的 next-token prediction，并把"推理未来"和"渲染像素"这两件事彻底解耦**


GAIA-1（Wayve, 2023）这篇文章的核心可以一句话概括：**把自动驾驶的"世界模型"问题重新表述成 LLM 式的 next-token prediction，并把"推理未来"和"渲染像素"这两件事彻底解耦**。下面按它的设计逻辑拆开讲。

**整体架构：两段式分工**
First, we partition the model into two components: the world model and the video diffusion decoder. The world model reasons about the scene’s high-level components and dynamics, while the diffusion model takes on the responsibility of translating latent representations back into high-quality videos with realistic detail.

模型分成两个独立组件，这是全文最关键的设计决定：

1. _World Model_（6.5B，自回归 Transformer）—— 在离散 token 空间里预测未来，负责场景的高层结构与动力学。它不碰像素。
2. _Video Diffusion Decoder_（2.6B，视频扩散模型）—— 只负责把 world model 吐出的 image token 翻译回高分辨率、时间一致的像素视频，并做时间上采样。

images, text and actions are encoded as a sequence of tokens.The world model is an auto regressive transformer that predicts the next image token conditioned on past image, text, and action tokens. Finally, the video decoder maps the predicted image tokens back to the pixel space, at a higher temporal resolution.

**三种模态如何进入同一个 token 序列**

每个时间步把三种输入编码到共享的 $d=4096$ 空间：

- _图像_：用一个预训练 VQ 自编码器把每帧离散化。下采样因子 $D=16$，每帧 $n = \frac{H}{D}\times\frac{W}{D} = 18\times32 = 576$ 个 token，码本大小 $K=8192$。
- _文本_：T5-large 编码，$m=32$ 个 token。
- _动作_：只有速度和曲率两个标量，$l=2$。

# Encoding Video, Text and Action

**Image tokens.** Each image frame of a video is represented as discrete tokens. To achieve this, we use a **pre-trained image tokenizer** for discretization (for details about the pre-training see Section 2.2). Formally, let us consider a sequence of $T$ images $(x_1, \ldots, x_T)$, where each image $x_t$ in this sequence is discretized into $n = 576$ discrete tokens using the pre-trained image tokenizer. We obtain a sequence denoted as $(z_1, \ldots, z_T)$, where each $z_t = (z_{t,1}, \ldots, z_{t,n}) \in \mathbb{R}^n$ corresponds to $n = \frac{H}{D} \times \frac{W}{D}$ discrete tokens. Here, $H$ and $W$ represent the height and width of the input image, while $D$ denotes the **downsampling factor** of the image tokenizer. These discrete tokens are then mapped to a **$d$-dimensional space** via an embedding layer that is trained alongside the world model.

**Text tokens.** At each time step $t$, we incorporate information from both text and action. Textual input is encoded using the **pre-trained T5-large model** [24], resulting in $m = 32$ text tokens per time step. These tokens are mapped to a $d$-dimensional space through a linear layer that is trained in conjunction with the world model. This process yields a text representation denoted as $c_t = (c_{t,1}, \ldots, c_{t,m}) \in \mathbb{R}^{m \times d}$.

**Action tokens.** For actions, we consider $l = 2$ scalar values (representing **speed** and **curvature**). Each scalar is independently mapped to the $d$-dimensional space via a linear layer that is trained with the world model. Consequently, the action at time step $t$ is represented as $a_t = (a_{t,1}, \ldots, a_{t,l}) \in \mathbb{R}^{l \times d}$. For each time step, the input tokens are interleaved in the following order: text — image — action.

The final input of the world model is therefore $(c_1, z_1, a_1, \ldots, c_T, z_T, a_T)$. To encode the position of the input tokens, we use a **factorized spatio-temporal positional embedding**.

1. A **learnable temporal embedding** is shared across all the tokens of a given time step, i.e. there are $T$ temporal embeddings.
2. A **learnable spatial embedding** indicates the position of a token within a time step, i.e. there are $m + n + l = 610$ spatial embeddings ($m$ text tokens, $n$ image tokens, and $l$ action tokens) of dimension $d = 4096$.


原文三句:
- 图像:embedding layer "trained **alongside** the world model"
- 文本:linear layer "trained **in conjunction with** the world model"
- 动作:linear layer "trained **with** the world model"
alongside / in conjunction with / with —— 这三个是同义反复,都只是在说同一件事:**这一层和 world model 端到端联合训练,不是冻结的预训练权重**。作者只是写作时换了词,没有任何技术区别。

**真正该区分的是两层结构**
每个模态其实有两级编码,容易混:
1. _上游重型编码器(冻结)_:图像用预训练 VQ tokenizer,文本用预训练 T5-large。训 world model 时这些是**冻住的**。
2. _下游投影层(联合训练)_:把上游输出映射到 world model 的共享 $d=4096$ 空间。"alongside / in conjunction" 指的就是这一层——它跟着 world model 一起学。

所以三句话的共同意思是:**上游编码器冻结,下游投影层联合训练**。

**图像 vs 文本投影层的真实区别:embedding layer 还是 linear layer**

这才是有内容的差异,而且原文确实用了不同的词:

- _图像 → embedding layer_。因为 VQ tokenizer 输出的是**离散 token id**(码本里的整数索引,$K=8192$),要把整数映射成向量,必须用查表式的 embedding(就是 LLM 里的词向量表 `nn.Embedding`)。
- _文本 → linear layer_。T5-large 输出的是**连续特征向量**,已经是实数向量了,只需要一个线性变换换一下维度。
- _动作 → linear layer_。2 个标量(速度、曲率),同样是连续值,线性映射即可。

一句话总结:**"embedding 还是 linear" 的区别来自输入是离散还是连续**(图像离散用查表,文本/动作连续用线性投影);而 "alongside 还是 in conjunction" 只是同义换词,都表示联合训练。你读到的两个词不构成区别。






每个时间步按 **text–image–action** 的顺序交错，整段序列是 $(c_1, z_1, a_1, \dots, c_T, z_T, a_T)$。位置编码是分解式的时空 embedding：一个共享给整个时间步的时间 embedding（共 $T$ 个），加一个步内的空间 embedding（共 $m+n+l=610$ 个）。

**DINO 蒸馏——这是 tokenizer 的精髓**
普通 VQ-GAN 的 token 会被高频纹理主导,对世界建模没用。作者在量化特征上加了一个 cosine similarity 损失,强迫 token 去回归预训练 DINO 的特征(论文图 3:同一语义类——车、路、天空——的 token embedding 变得相似)。本质是**把压缩方向从"保留高频信号"扭向"保留语义"**,让下游自回归模型面对的输入空间更简单、更可学。tokenizer 的总损失 = 图像重建(L1/L2/perceptual/GAN)+ 量化损失 + DINO inductive bias 损失。

**World Model 的训练目标**

就是标准的 causal next-token,只不过 token 是图像 token:

$L_{\text{world model}} = -\sum_{t=1}^{T}\sum_{i=1}^{n}\log p(z_{t,i}\mid z_{<t}, z_{t,j<i}, c_{\le t}, a_{<t})$

训练时随机 dropout 条件 token,于是同一个模型能做三件事:无条件生成、动作条件生成、文本条件生成(比例 20/40/40)。另外把视频从 25Hz 降采样到 6.25Hz,让序列长度可控(总长 $T\times(m+n+l)=26\times610=15860$),丢失的帧率后面由解码器做时间超分补回来。

**采样策略与困惑度诊断(图 6)**

这是个很漂亮的实证分析,和 Holtzman 的 neural text degeneration 一脉相承:

- argmax → 困惑度极低 → 没有多样性 → 画面陷入重复循环。
- 全分布采样 → 偶尔抽到不可靠长尾 → 把模型推出分布外。
- top-k(k=50)→ 困惑度分布最接近真实 token。

所以最终用 top-k。

**Classifier-free guidance 的调度**

文本条件用 CFG 增强对齐:

$l_{\text{final}} = (1+t),l_{\text{conditioned}} - t,l_{\text{unconditioned}}$

把无条件 logits 换成另一个 prompt 的 logits 就实现 negative prompting。一个细节:guidance scale 要**同时在 token 维度和 frame 维度上调度**——token 上线性递减(部分 token 强遵循 prompt、部分保持多样性),frame 上用 cosine decay(避免后续帧 guidance 累积爆掉)。

**Video Decoder 的几个工程选择**

3D U-Net + 分解的空间/时间注意力。用 v-parameterization(避免颜色漂移、保持长程一致)。多任务联合训练(图像生成、视频生成、自回归解码、视频插值,各等概率采样),并对条件 token 做 $p=0.15$ 的 dropout 防止过度依赖。损失是带噪声预测的加权 L1+L2:

$L_{\text{video}} = \mathbb{E}_{\epsilon,t'}\big[,\lVert \epsilon_\theta(x^{t'}, t', z, m) - \epsilon \rVert_2^2,\big]$

推理时还在"逐帧当图像去噪"和"整段当视频联合去噪"之间做加权($w=0.5$, $p=0.25$),平衡 token 信息保真度 vs 时间一致性。一个有意思的经验发现:**从序列末尾往前做自回归解码**,horizon 上的物体更稳、闪烁更少。

**Scaling laws**

既然 task 被改写成 next-token,LLM 的幂律自然迁移过来。用比最终模型小 10x–10000x(0.65M–650M)的模型拟合 $f(x)=c+(x/a)^b$,就能高精度预测 6.5B 模型的最终 cross-entropy。compute 估计仍用 $C=6N$。结论是数据和算力都还远没到天花板。

**涌现能力——为什么这篇值得关注**

这部分和你关心的 world-model-as-policy / 神经模拟器方向直接相关:

- 学到高层结构与场景动力学(红绿灯、让行规则),不只是统计记忆。
- 3D 几何理解:能正确表现减速带引起的 pitch/roll 变化。
- 因果性:强制 ego 车右转,对向来车会做出避让机动(图 13 最后一行)。
- **分布外外推**:训练集是专家驾驶数据,从没有"开出车道"的样本,但你给它 OOD 动作(strong left/right),它能生成几何正确的后果。这说明它把 ego 动力学和环境解耦了。
- 能纯靠 imagination 生成几分钟的稳定长视频。

最后作者明确点出 GAIA-1 作为 **neural simulator** 的价值:闭环探索、model-based policy learning、生成无限的对抗样本用于训练/验证。已知局限是自回归生成还不能实时(但可并行多样本)。






先澄清一个容易混的点:**这里的"DINO 蒸馏"和 DINO 论文本身的"自蒸馏"不是同一件事**。

DINO(Caron et al., 2021,_self-**di**stillation with **no** labels_)本身是一种自监督预训练算法。而 GAIA-1 里说的"DINO distillation",指的是把**一个已经训练好、冻结的 DINO 模型当作 teacher**,把它的特征知识蒸馏进 VQ tokenizer 的 token 里。换句话说,DINO 在这里只是个"语义特征提供者",GAIA-1 并没有重新跑 DINO 的自蒸馏流程。所以更准确的叫法是 _feature distillation from a frozen DINO teacher_。这个思路直接借鉴自 BEiT v2(论文引用 [29],_Masked Image Modeling with Vector-Quantized Visual Tokenizers_)。

**为什么 DINO 的特征"有语义"**

要理解为什么值得蒸馏它,得先知道 DINO 特征好在哪。DINO 的训练机制:

- 一个 student 网络和一个 teacher 网络,结构相同,**teacher 的权重是 student 的 EMA(动量编码器)**。
- multi-crop:student 看全局大裁剪 + 局部小裁剪,teacher 只看全局裁剪。
- student 被训练去匹配 teacher 在一组 prototype 上的输出分布(softmax 后的交叉熵)。
- 用 centering(防止某一维独大)+ sharpening(低温度,防止塌成均匀分布)来避免 collapse。
- 没有负样本、没有对比损失、没有标签。

它最出名的涌现性质是:**ViT 的 [CLS] attention 会自发地分割出物体**,patch 特征天然按语义类聚类——不需要任何分割标注,k-NN 就能做分类/分割。也就是说 DINO 特征空间的几何结构本身就编码了"这是车 / 这是路 / 这是天空"。这正是 GAIA-1 想白嫖的东西。

**GAIA-1 把 DINO loss 加在哪里**

image tokenizer 是个 VQ 自编码器(全卷积 2D U-Net)。流程是:编码器 $E_\theta$ 输出连续特征 $z_e(x)$ → 在可学习码本里做最近邻查找 → 量化特征 $z_q(x)$ 与离散 token。它的训练损失有三组:

1. _重建损失_:解码器重建图像,$L_1+L_2+$ perceptual $+$ GAN。(注意:解码器只在训练 tokenizer 时用,最终只有编码器 $E_\theta$ 进入 GAIA-1。)
2. _量化损失_:VQ-VAE 的 codebook loss + commitment loss, $L_{\text{VQ}} = \lVert \text{sg}[z_e(x)] - e \rVert_2^2 + \beta\lVert z_e(x) - \text{sg}[e]\rVert_2^2$ 再加上 improved-VQGAN(引用 [34])的 trick——码本做低维线性投影 + L2 归一化,目的是**提高码本利用率**(避免大量 dead codes)。
3. _Inductive bias loss(即 DINO 蒸馏)_:把量化后的图像特征,通过一个投影对齐到对应空间位置的 DINO 特征,用 **cosine similarity 损失**拉近,权重 $\lambda_{\text{DINO}}=0.1$。形式上大致是 $L_{\text{DINO}} = 1 - \cos\big(\phi(z_q(x)),, f_{\text{DINO}}(x)\big)$

这里有个空间对齐的细节:DINO 的 patch 特征网格和 VQ token 网格($18\times32$,$D=16$)要在每个空间位置一一对应,所以是逐位置回归 DINO 特征,而不是只匹配一个全局向量。

**为什么非要这么做——这是全篇 tokenizer 的精髓**

关键在于一个张力:**重建损失和 GAN 损失会把 token 往高频纹理方向拉**。这两个目标奖励"像素级还原",所以裸 VQ-GAN 的 token 容易被纹理、光照、边缘这类高频信号主导——两块语义完全不同但纹理相近的 patch 可能共用 token,反之亦然。

可下游的 world model 要在这些 token 上做自回归 next-token prediction。如果 token 是按纹理组织的,那"预测下一个 token"就退化成在预测高频噪声;如果 token 是按语义组织的,这个任务就更接近**在物体/场景结构层面推理未来**。论文原话就是要让 world model 的输入空间"更易于组合、更少被高频信号支配,后者会显著拖慢学习"。DINO 蒸馏就是把压缩方向从"保真高频"硬扭向"保留语义"的那只手。

图 3 是定性证据:把 token embedding 的前 3 个 PCA 主成分映成 RGB。裸 VQ-GAN 的 token 在这个空间里看起来杂乱;而 DINO 蒸馏后,**同一语义类(车、路、天空)的 token embedding 颜色相近**,说明 embedding 空间真的被组织成了语义簇。

**为什么是 DINO,而不是 CLIP 或监督特征**

因为 DINO 是自监督的——不需要任何人工标注就能在 4700 小时驾驶数据这种规模上提供语义先验,从而**保持整个 GAIA-1 流水线端到端 self-supervised**(这是论文反复强调的卖点)。换成监督 backbone 就要标注,换成 CLIP 会引入文本-图像对齐的偏置;DINO 的纯视觉语义结构在这里更"干净"。

**和你关心的方向的联系**

从你研究 representation learning / score function / 可解释性的角度,这件事的本质是:**在量化瓶颈处主动塑造离散 latent manifold 的几何,使它对齐到一个语义 manifold**。它是 feature distillation(回归 teacher 的中间特征),而不是 logit distillation(匹配输出分布)。

对自回归世界模型而言,效果类似于"换了一套词表的基":让语义近邻在码本里也成为近邻——这跟 LLM 里 tokenization 质量决定下游难度是同构的道理。你做 VLA / diffusion policy 时如果要设计自己的离散动作或观测 token,这是一个很值得借鉴的范式:用一个带强归纳偏置的冻结 teacher,在量化处加一项 cosine 对齐,把 codebook 的语义结构"种"进去。

要不要我再展开一下 VQ 那边的 codebook 利用率问题(L2 归一化 + 投影为什么能缓解 codebook collapse)?这块和 DINO 蒸馏其实是配合着一起保证 token 质量的。


这句话其实压缩了三个独立的点,拆开就清楚了。

**两个网络,结构相同,但权重不共享**

DINO 同时维护两份参数:student $\theta_s$ 和 teacher $\theta_t$。网络结构(比如同一个 ViT)完全一样,但它们是**两套独立的权重值**。同一张图(经过不同裁剪)分别喂给两者,得到两个输出分布,训练目标是让 student 的输出去匹配 teacher 的输出。

**只有 student 被反向传播,teacher 不吃梯度**

关键的不对称:loss 的梯度**只更新 student**。teacher 这一侧加了 stop-gradient,它从不通过反向传播学习。那 teacher 的权重怎么变?靠下面这条规则。

**teacher 的权重是 student 的 EMA(指数移动平均)**

每一步 student 做完梯度更新后,teacher 按一个凸组合缓慢跟进:

$\theta_t \leftarrow \lambda,\theta_t + (1-\lambda),\theta_s$

其中 $\lambda$ 非常接近 1(DINO 里大约从 0.996 用 cosine schedule 升到 1)。因为 $\lambda$ 很大,teacher 每一步只吸收一丁点 student 的新权重,所以它是 student **过去轨迹的一个平滑、滞后的版本**。这就是"动量编码器(momentum encoder)"这个名字的来历——和 MoCo、BYOL、Mean Teacher 里用的是同一套机制。

**为什么要这么绕,而不直接让网络匹配自己的输出**

这是整件事的核心。如果你让一个网络直接预测"自己当下的输出",目标和预测器同速移动,动力学很不稳定,极易塌成平凡解(所有输入输出同一个常数)。EMA 解决的正是这点,它带来三个好处:

1. _稳定的目标_。teacher 滞后、平滑,提供的预测目标比 student 自己当下的输出更平稳,student 是在追一个"略微过时但更可靠"的自己。
2. _沿训练轨迹的集成(temporal ensembling)_。权重的移动平均相当于把训练过程中很多个 checkpoint 平均起来(类似 Polyak averaging),经验上得到的特征质量比任何单一 checkpoint 都好。所以 teacher 通常比 student "更强",拿它当老师才合理。
3. _自举(bootstrapping)而不崩溃_。teacher 给目标 → student 变好 → EMA 让 teacher 慢慢吸收这些改进 → teacher 给出更好的目标 → …… 这个正反馈循环,配合 teacher 那侧的 centering + sharpening(防止两种 collapse),让表征能在**没有标签、没有负样本**的情况下一步步自己变好。

一个直观比喻:teacher 就像你"最近一段时间的平均水准",student 是"此刻正在努力的你"。让此刻的你去对齐那个更稳、更平滑的平均自我,而不是对齐瞬息万变的当下,学习过程才不会发散或塌缩。





常用,但要看场景——它几乎是"想对非文本模态做自回归建模"时的默认工具,而在扩散生成那一支里反而不是主流。

**VQ tokenizer 很常见的地方**

凡是想把图像/视频/音频塞进 next-token prediction 框架的,基本都用 VQ:

- _自回归图像/视频生成_:VQGAN + Transformer(Taming Transformers)、DALL·E 1(discrete VAE)、Parti、MaskGIT、MAGVIT(视频)、VideoGPT。
- _统一多模态 LLM_:想让图像和文本共享一个 token 序列、用同一套交叉熵训练的模型(如 Chameleon 这类)。
- _世界模型_:GAIA-1 正是这一类——目的就是复用 LLM 的整套 next-token 机制。
- _神经音频编解码 / 语音_:这块 VQ 是绝对主流,SoundStream、EnCodec 用的 residual VQ(RVQ)是标配,上层的 AudioLM、MusicLM、VALL-E 都建在它上面;wav2vec 2.0、HuBERT 也用离散单元。

**VQ 不是默认、用连续 latent 的地方**

- _扩散生成_:Stable Diffusion 这类 latent diffusion 用的是**连续的 KL-VAE latent**,不是离散 VQ。
- _部分新的自回归图像模型_:像 GIVT、MAR(在连续 token 上用 diffusion loss 做自回归)就是为了**绕开 VQ 的缺点**而改用连续 token。

**为什么这么分——核心权衡**

选 VQ 的唯一最大理由:它给你一个**离散词表**,于是 LLM 的全套工具(交叉熵、采样、top-k、scaling law)直接拿来就能用,和 Transformer 天然契合。代价是:

- 量化是有损的信息瓶颈,重建质量被码本上限卡死;
- **codebook collapse / 利用率低**(GAIA-1 用 L2 归一化 + 低维投影来缓解的就是这个);
- argmax 不可导,要靠 straight-through 估计梯度。

所以一句话:**要做"序列建模 + 离散采样"就用 VQ;要做扩散就多半用连续 latent。** GAIA-1 选 VQ 是被它"world model = next-token"的范式逼出来的必然结果。

**和你的方向**

这点对你做 VLA / diffusion policy 直接相关:动作离散化是同一套思路。RT-1/RT-2 把动作分 bin 离散化,也有工作用 VQ 把动作块(action chunk)压成离散 token 再做自回归。要不要我顺带讲一下动作 token 化里 VQ vs 连续(扩散头)这两条路线在策略学习上的取舍?那跟你 diffusion policy 的工作会更贴。