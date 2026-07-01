### A review of learning-based dynamics models for robotic manipulation 
This article focuses on models of environmental dynamics external to the robot.
Learning-based dynamics models predict how the world evolves in response to actions.

**the perspective of this paper**

The paper argues that model structure should reflect physical structure. Particle systems are modeled with GNNs (capturing spatial equivariance), rigid bodies with SE(3)-equivariant networks, and multi-object systems with object-centric representations. The inductive bias is derived from physical symmetries, leading to higher sample efficiency and stronger generalization.

This is a **bottom-up design philosophy**: first understand the structure of the problem, then choose a model that respects that structure.

**The Transformer/VLA perspective**
Transformers contain essentially no built-in physical inductive bias. They are general-purpose sequence processors that compensate for the lack of structural priors through **scale and data**.
The logic behind Vision-Language-Action (VLA) models is:
- Rather than carefully engineering inductive biases for every physical process, use sufficiently large models and sufficiently large datasets, allowing the network to discover useful structure on its own.
- Language serves as the task specification interface, vision provides perceptual input, and actions are generated as outputs, all within a unified sequence-to-sequence framework.
- Generalization across tasks and environments comes primarily from **pretrained representations**, rather than manually encoded symmetries.




Physics-based models (3, 4) generalize well but rely on full-state information, which is often unattainable in real- world manipulation tasks. Learning-based models offer an alternative by learning predictive dynamics directly from raw sensory data, capturing hard-to- model factors (5, 6), reasoning about uncer- tainty (6–8), and accelerating high-precision simulations too slow for real-time control (9, 10). 


learning-based models face a fundamental challenge: designing inductive biases that ensure sample efficiency and generalization This is especially acute in robotics — real-world data collection is expensive, and open-world environments span vast, high-dimensional state spaces.

To remain tractable, effective models compress information into compact state representations and incorporate structural priors that guide how the model processes and relates states. Yet this compactness introduces a tension: constraining the state space improves generalization but risks limiting model expressiveness or making state estimation harder. The right design choices depend on the task at hand, the complexity of the environment, and sensory modalities . 


**Inductive bias** is a prior assumption imposed on a model that restricts its hypothesis space. Examples include:
**Convolutional Neural Networks (CNNs)** assume that features exhibit **translation invariance** and **locality**.
**Graph Neural Networks (GNNs)** assume that interactions between objects are **permutation invariant**.
 **Lagrangian Neural Networks (LNNs)** assume that the underlying system obeys **energy conservation**.

**Sample efficiency** refers to the ability to learn an equally good model from fewer training examples. Inductive bias improves sample efficiency because it **narrows the hypothesis space**, reducing the amount of data needed to rule out incorrect solutions.

**Generalization** refers to a model's ability to perform well on data outside the training distribution. A well-chosen inductive bias reflects the true structure of the physical world, allowing the learned model to remain valid in novel situations and environments.



 object repositioning 
13.L. Manuelli, Y. Li, P. R. Florence, R. tedrake, “Keypoints into the future: Self-supervised correspondence in model-based reinforcement learning,” in Proceedings of the 2020 Conference on Robot Learning, J. Kober, F. Ramos, c. tomlin, eds., vol. 155 of Proceedings of Machine Learning Research (PMLR, 2020), pp. 693–710. 

Z. Xu, J. wu, A. Zeng, J. b. tenenbaum, S. Song, “DensePhysnet: Learning dense physical object representations via multi-step dynamic interactions,” in Proceedings of Robotics: Science and Systems, A. bicchi, H. Kress- Gazit, S. Hutchinson, eds. (RSS Foundation, 2019). 

 P. Agrawal, A. nair, P. Abbeel, J. Malik, S. Levine, “Learning to poke by poking: experiential learning of intuitive physics,” in Advances in Neural Information Processing Systems 29 (NeurIPS 2016), D. Lee, M. Sugiyama, U. Luxburg, i. Guyon, R. Garnett, eds. (curran Associates, 2016).
deformable object handling (16–19),

H. Shi, H. Xu, S. clarke, Y. Li, J. wu, “Robocook: Long-horizon elasto-plastic object manipulation with diverse tools,” in Proceedings of the 7th Conference on Robot Learning, J. tan, M. toussaint, K. Darvish, eds., vol. 229 of Proceedings of Machine Learning Research (PLMR, 2023), pp. 642–660.

 H. Shi, H. Xu, Z. Huang, Y. Li, J. wu, Robocraft: Learning to see, simulate, and shape elasto- plastic objects in 3D with graph networks. Int. J. Rob. Res. 43, 533–549 (2024). 
 
 A. Longhini, M. Moletta, A. Reichlin, M. c. welle, D. Held, Z. erickson, D. Kragic, “eDo- net: Learning elastic properties of deformable objects from graph dynamics,” in 2023 IEEE International Conference on Robotics and Automation (ICRA) (ieee, 2023), pp. 3875–3881. 
 
W. Yan, A. vangipuram, P. Abbeel, L. Pinto, “Learning predictive representations for deformable objects using contrastive estimation,” in Proceedings of the 2020 Conference on Robot Learning, J. Kober, F. Ramos, c. tomlin, eds., vol. 155 of Proceedings of Machine Learning Research (PMLR, 2020), pp. 564–574.



multimodal perception (20, 21), 


B. Ai, S. tian, H. Shi, Y. wang, c. tan, Y. Li, J. wu, “RoboPack: Learning tactile-informed dynamics models for dense packing,” in Proceedings of Robotics: Science and Systems, D. Kulic, G. venture, K. bekris, e. coronado, eds. (RSS Foundation, 2024). 

S. tian, F. ebert, D. Jayaraman, M. Mudigonda, c. Finn, R. calandra, S. Levine, “Manipulation by feel: touch-based control with deep predictive models,” in 2019 International Conference on Robotics and Automation (ICRA) (ieee, 2019), pp. 818–824.


multiobject interaction (14, 22, 23)

Z. Xu, J. wu, A. Zeng, J. b. tenenbaum, S. Song, “DensePhysnet: Learning dense physical object representations via multi-step dynamic interactions,” in Proceedings of Robotics: Science and Systems, A. bicchi, H. Kress- Gazit, S. Hutchinson, eds. (RSS Foundation, 2019).

Y. wang, Y. Li, K. D. campbell, L. Fei-Fei, J. wu, “Dynamic-resolution model learning for object pile manipulation,” in Proceedings of Robotics: Science and Systems, K. bekris, K. Hauser, S. Herbert, J. Yu, eds. (RSS Foundation, 2023). 

D. Driess, Z. Huang, Y. Li, R. tedrake, M. toussaint, “Learning multi-object dynamics with compositional neural radiance fields,” in Proceedings of the 6th Conference on Robot Learning, K. Liu, D. Kulic, J. ichnowski, eds., vol. 205 of Proceedings of Machine Learning Research (PMLR, 2022), pp. 1755–1768.


![[Pasted image 20260602092514.png]]**three levels of state abstraction** for dynamics modeling, and motivates why structured representations matter.

**The three representations:**

- **(A) Particles:** the object is discretized into a large number of small particles, capturing fine-grained geometry and deformation. Suited for continuous or deformable materials — cloth, sand, soft bodies.
- **(B) Keypoints:** only a sparse set of structurally significant points are retained (e.g. joint positions, object corners), discarding redundant geometry while preserving kinematically relevant information.
- **(C) Object-centric representations:** the scene is decomposed into discrete object entities, each with its own state, and interactions between objects are modeled explicitly. Suited for multi-object manipulation scenarios.


**Learning-Based Dynamics Models: Framework and Motivation**
**Formal setup**
The system is modeled as a POMDP. At each timestep, the agent observes $o_t \in \mathcal{O}$ instead of the true state $s_t \in \mathcal{S}$, takes action $a_t = \pi(o_t)$, and the environment transitions as $s_{t+1} = \mathcal{T}(s_t, a_t)$. The objective is to find a policy minimizing cumulative cost over horizon $H$:
$$\min_\pi \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^{H} c(s_t, a_t) \right]$$
**Three-module pipeline**

| Module                       | Function                                                                                                                          | Key challenge                                            |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| Perception $g$               | Estimates $s_t$ from observation history $o_{0:t}, a_{0:t-1}$,simplifies to $s_t = g(o_t, a_{t–1})$ in fully observable settings. | What to include in $s_t$: minimal yet sufficient         |
| Dynamics $\hat{\mathcal{T}}$ | Predicts $s_{t+1}$ from $s_t, a_t$                                                                                                | Architecture must match state representation structure   |
| Control $\pi$                | Generates actions to minimize cost                                                                                                | Planning vs. policy learning; position vs. force outputs |


The three modules are coupled: the choice of $s_t$ simultaneously constrains what $g$ must estimate, what $\hat{\mathcal{T}}$ must predict, and what $\pi$ operates on.

**Why learned dynamics**
Classical physics-based models fail not because their equations are wrong, but for three concrete reasons:
1. **Unmodeled effects** — frictional contact, actuator drift are hard to express analytically
2. **Missing latent factors** — temperature and other unobservable variables undermine accuracy even with correct parameters
3. **Error accumulation** — small state estimation errors compound over a rollout horizon
Learned dynamics models address this by fitting $\hat{\mathcal{T}}$ directly to interaction data, absorbing these effects implicitly. They can also compensate for state estimation errors and bypass the need for accurate system identification.

learned models are also end-to-end differentiable, enabling gradient- based planning, control, and online adaptation. Some studies found that learned models offer smoother gradients than analytical solvers (36) and can be more computationally efficient, especially for nonrigid systems (37).

**Central design tension**
We view $s_t$ as a unified representation of all task-relevant information inferred from raw sensory data.
 A central challenge lies in defining $s_t$ to capture minimal yet sufficient in- formation for manipulation. 




![[Pasted image 20260602095825.png|697]]

Fig. 2. Robotic manipulation using learning-based dynamics models.
A  A dynamics model is trained on interaction data. The perception model extracts state representations $s_t$ from observations $o_t$. Dynamics are learned in a self-supervised fashion.
B  The learned dynamics model is applied for downstream control, either by evaluating action trajectories $\{a^i_{0:H}\}^{N}_{i=0}$ for planning or by generating interaction data $\{s^i_{0:H}, a^i_{0:H}\}^{N}_{i=0}$ for policy learning.


**STATE REPRESENTATIONS**
1. **Raw observations**, such as pixels encoding RGB (red, green, blue), depth, or force fields.
2. **Latent representations** Compress raw sensory input (images, point clouds) into a low-dimensional vector via an encoder. Compact and learnable end-to-end, but the encoding is opaque — there's no explicit geometry inside it. The model doesn't "know" where things are in 3D space, it just has a vector of numbers.

 3. **Particle representations** To fix the geometry problem, discretize the scene into explicit 3D particles. Now the model has actual spatial structure — surfaces, volumes, deformations are all representable. But the cost is resolution: for a simple pick-and-place task, tracking 10,000 particles is wasteful. You're carrying far more information than the task requires.

4. **Keypoints** Downsample further — instead of all particles, keep only a sparse set of task-relevant points. A robot arm might only need the positions of a few corners or joints, not the full geometry. More efficient, but now you've lost the continuous geometric structure. Also, keypoints are an unstructured set — there's no notion of "this point belongs to that object."

5. **Object-centric representations** The key insight: humans don't perceive scenes as bags of points, they perceive _objects_. An object-centric representation explicitly groups elements into discrete entities and models interactions _between_ objects rather than between individual points. This adds a layer of relational structure that none of the previous representations have.

![[Pasted image 20260602102830.png]] A spectrum of state representations with varyin 2025_scirobotics.adt1497.pdf
'A. L. Onishchik E. B.Vinb'$'\n''Lie Groups'$g structural priors. State representations in dynamics models range from unstructured (pixels and latent) to structured (particles, keypoints, and object-centric). increasing structure introduces stronger priors and abstraction, enabling better generalization but requiring more complex state estimation. the “Swiss roll” illustration for latent states was inspired by tenenbaum et al. (132) and created using Python.

Pixel-based models autoregressively generate future observations, whereas latent-state models predict in the abstract latent space. This projection introduces inductive biases by assuming that the state space admits a compact and smooth parameterization. This can substantially enhance learning efficiency and generalization by filtering out irrelevant variations.



Learning dynamics models in pixel space
action- conditioned video prediction


H. t. Suh, R. tedrake, “the surprising effectiveness of linear models for visual foresight in object pile manipulation,” in Algorithmic Foundations of Robotics XIV: Proceedings of the Fourteenth Workshop on the Algorithmic Foundations of Robotics, S. M. Lavalle, M. Lin, t. ojala, D. Shell, J. Yu, eds., vol. 17 of Springer Proceedings in Advanced Robotics (Springer, 2021). 42. A. Gupta, S. tian, Y. Zhang, J. wu, R. Martín-Martín, L. Fei-Fei, Maskvit: Masked visual pre-training for video prediction, paper presented at the eleventh international conference on Learning Representations, Kigali, Rwanda, 1 to 5 May 2023. 43. A. Hu, L. Russell, H. Yeo, Z. Murez, G. Fedoseev, A. Kendall, J. Shotton, G. corrado, GAiA-1: A generative world model for autonomous driving. arXiv:2309.17080 [cs.cv] (2023). 44. Y. Du, M. Yang, P. Florence, F. Xia, A. wahid, b. ichter, P. Sermanet, t. Yu, P. Abbeel, J. b. tenenbaum, L. Kaelbling, A. Zeng, J. tompson, video language planning, paper presented at the twelfth international conference on Learning Representations, vienna, Austria, 5 to 11 May 2024. 45. M. Yang, Y. Du, K. Ghasemipour, J. tompson, D. Schuurmans, P. Abbeel, Learning interactive real-world simulators, paper presented at the twelfth international conference on Learning Representations, vienna, Austria, 5 to 11 May 2024. 46. Y. Du, S. Yang, b. Dai, H. Dai, o. nachum, J. tenenbaum, D. Schuurmans, P. Abbeel, “Learning universal policies via text-guided video generation,” in Advances in Neural Information Processing Systems 36 (NeurIPS 2023), A. oh, t. naumann, A. Globerson, K. Saenko, M. Hardt, S. Levine, eds. (curran Associates, 2023), pp. 9156–9172. 47. R. Hoque, D. Seita, A. balakrishna, A. Ganapathi, A. K. tanwani, n. Jamali, K. Yamane, S. iba, K. Goldberg, visuoSpatial Foresight for physical sequential fabric manipulation. Auton. Robots 46, 175–199 (2022). 48. S. Xue, S. cheng, P. Kachana, D. Xu, “neural field dynamics model for granular object piles manipulation,” in Proceedings of the 7th Conference on Robot Learning, J. tan, M. toussaint, K. Darvish, eds., vol. 229 of Proceedings of Machine Learning Research (PLMR, 2023), pp. 2821–2837. 49. o. Rybkin, K. Pertsch, K. G. Derpanis, K. Daniilidis, A. Jaegle, Learning what you can do before doing anything, paper presented at the Seventh international conference on Learning Representations, new orleans, LA, 6 to 9 May 2019. 50. K. Schmeckpeper, A. Xie, o. Rybkin, S. tian, K. Daniilidis, S. Levine, c. Finn, “Learning predictive models from observation and interaction,” in Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XX, A. vedaldi, H. bischof, t. brox, J.-M. Frahm, eds., vol. 12365 of Lecture Notes in Computer Science (Springer, 2020), pp. 708–725.