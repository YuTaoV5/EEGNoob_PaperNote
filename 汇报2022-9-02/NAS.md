# NAS
## Introduction
在现今电脑视觉的领域当中，CNN 已经可以达到非常高甚至超越人类的精确度，但随之而来的代价就是追求更深更大的网路所造成的高复杂度（例如:大量的记忆体需求，大量的运算），因此对于在有限资源的硬体设备（例如：手机，嵌入式系统）中应用产生了巨大的挑战，因为我们总不可能在每一台手机每一台监视器，都放一张 1080Ti 或是 2080Ti，所以现今的网路需求除了对于精确度的要求，也慢慢开始注重效能的优化。
然而，设计出一个能够兼顾准确度以及效能的网路是很困难的。
要设计一个好的网路，首先会遇到两个问题，分别是 intractable search space 和 non transferable optimality。

### Intractable Search Space
Search space 的意思是在我们搜索出一个网路的时候，我们的搜索的空间，也就是所有可能的选择所组合成的空间。
以大家熟悉的 VGG16 为例，其中并没有任何较为复杂的设计（例如：residual connection），单纯为 13 层的 convolutional layers，假设在每一层 kernel size 的选择皆是 {1, 3, 5}，单是这样可能构成的选择便有 3¹³ 次方种选择。且训练一个 CNN 往往会花费许多时间，更不用说要从这么多的选择中要挑出一个最好的架构了，毕竟我们不可能实际的训练这 3¹³ 种网路吧！

### Non transferable optimality
而对于不同的装置，由于其不同的硬体架构，因此对于同样的一个运算，也会有不同的速度，所以一个在 A 装置表现的好的网路架构，并不一定同样可以在 B 装置表现的好，也就是 CNN 的 non transferable optimality。
因此便有人开始进行 neural architecture search 的研究，目的是希望可以开法出一个方法，并且可以自动的探寻 search space 以获得最好的网路。

## What is Neural Network Search (NAS)?
简单来说，NAS 的目的就是希望可以有一套演算法或是一个框架能够自动的根据我们的需求找到最好的 neural architecture，而我们的搜索目标有可能会是根据 performance，或是根据硬体资源限制 (hardware constraints) 来进行搜索。
而 NAS 可以分成三个大部份，也就是 search space, search strategy 和 performance estimation strategy。

### Search Space
中文又称作为搜索空间，也就是我们在选择 neural architecture 时，我们所可以调整的所有选择。举例来说，kerne size, channel size, convolution type 以及 layer number 等等。

### Search Strategy
在给定的 search space 当中，我们要透过什么方式来搜索出最好的 neural architecture。举例来说，在搜索 hyperparameter 时最为大家熟悉的 grid search 以及 random search，或是 evolution algorithm (基因演算法) 等等。

### Performance Estimation Strategy
当我们从 search space 当中挑选出了一个 neural architecture，我们如何评估这个 neural architecture 是好还是坏的方式。举例来说，我们可以实际的训练每个 neural architecture 来获得实际的 top-1 accuracy，我们也可以训练少量的 neural architecture，并且将实际训练好的数据，用来训练一个额外的 accuracy predictor。

## Reinforcement Learning NAS
![Img](https://imgpool.protodrive.xyz/img/yank-note-picgo-img-20220903021202.png#pic_center%20=400x)
对于 NAS 这个 task 来说，其实最直觉的方法就是我不断的从 search space 当中取出不同的 neural architecture ，并且实际的训练之后来获得真正的 performance，藉著不断的重复这个动作，当我穷尽整个 search space 时，我理所当然的就可以得到这个 search space 当中最好的那个 neural architecture。
而也因为这个简单的概念，所以最早的 NAS 是基于 reinforcement learning 的方式来进行搜索，也就是透过一个 controller (agent) 不断的挑出 neural architecture，并且透过 trainer (environment) 实际的进行训练，而所得到的 performance (e.g., top-1 accuracy, FLOPs or latency) 会作为 reward 进而更新 controller，使得 controller 可以学习要如何才可以 sample 出可以让 reward 最好的 neural architecture。
但由于每次 controller 所挑出的 neural architecture，都需要经过 trainer 实际的进行训练，因此可想而知的是这会是一个非常耗费时间成本的一件事情，因此为了使得 trainer 不会花费过大量的时间，过去的 work 会利用叫小型的 dataset (e.g., CIFAR10 or CIFAR100) 或是训练较少的 epoch 数。
然而，尽管这个方法在过去皆达到了很好的结果，但是这个方法依旧是需要一般人所无法负荷的庞大成本 (例如， CVPR 2019 Google 所提出的 MnasNet，在同时使用 64 个 TPU 的情况下，依旧是需要搜索 4.5 天！)，因此现今越来越少 NAS 使用 RL 的方式来进行搜索。

## Progressive NAS
Progressive NAS 最主要的概念便是一层一层渐进式的方式决定在每一层要用什么样的选择。
![Img](https://imgpool.protodrive.xyz/img/yank-note-picgo-img-20220903021227.png#pic_center%20=400x)
而由于要决定每一层要用什么样的架构，我们首先需要知道所有选择的好坏，而实际的将所有种可能的架构 training，并根据结果来决定好坏会花费许多时间，因此会额外的训练一个 controller，而 controller 的目的就是预测当前这样的组合的准确率会是多少。
因此 Progressive NAS 主要可以分成几个大步骤：
1、从 l-layers 的 N 个可能的网路架构中，透过 controller 预测结果，并挑选出最好的 K 个网路架构。
2、将 K 个网路架构实际的训练并得到他们间的好坏，并利用这 K 个结果更新 controller，使其可以预测的更加准确，将最好的网路架构接上下一层的所有可能选择，建构出 l+1 layers的 N 个架构，并回到步骤 1。

## One-shot NAS
为了减少过去 NAS 所耗费的大量时间 (e.g., 实际训练每个 neural architecture 的时间)，One-shot NAS 透过将 search space 当中的所有 neural architectures 结合成为一个 over-parameterized 的 neural network，而这个巨大且错综复杂的网路又被称作为 supernet。下图为建构 Supernet 的范例，假如在每层当中，我们总共有三种 candidate blocks 可以选择，则在 supernet 当中每层里面便会同时有三个不同 candidate blocks。
而这样建构的好处在于当 supernet 训练完毕之后，透过 activate supernet 当中不同的 candidate blocks，我们可以进而 approximate 在 search space 当中的任何一个 neural architecture (如下图最右边的部分)。因此 one-shot NAS 之所以称为 “one-shot” 的原因在于，我们只需要训练一个 neural network (supernet) ，便可以借此评估整个 search space 当中的任一个 neural architecture。
![Img](https://imgpool.protodrive.xyz/img/yank-note-picgo-img-20220903021249.png#pic_center%20=400x)
而在 One-shot NAS 当中，根据训练 supernet 以及搜索的不同，又可以分为两种，分别为 Differentiable NAS 以及 Single-path NAS。
想要实际实作看看 one-shot NAS 的读者们，欢迎参考这篇文章！

## Differentiable NAS
Differentiable NAS (DNAS) 希望可以透过 gradient descent 的方式来进行搜索。然而，由于 neural architectures 本身是离散的，因此我们没有办法直接对 neural architectures 微分并且计算梯度。因此 DNAS 使用了一个额外的可微分参数，我们称作为 architecture parameters，目的是希望透过这个可以微分的参数来学习在 search space 当中好的 neural architecture 的分布。
下图为 supernet 当中的某一层 layer，而 DNAS 首先对于每个 candidate block 皆会给予一个 architecture parameter，而这个值通常会是随机初始的 (如图中的 0.2, 0.7, 和 0.1)，因此对于这层 layer output 的计算方式，便会是每个 candidate block 本身的 output 和 architecture parameters 进行 weighted sum (如左图中的数学式)，透过这样的方式，我们便计算出 architecture parameters 的梯度，并且更新，而当 architecture parameters 更新完毕之后，便可以将之作为 “candidate blocks 之间重要程度的依据” ，进而直接对其做 argmax 并 sample 出搜索到的 architecture (如下图的右边)。
![Img](https://imgpool.protodrive.xyz/img/yank-note-picgo-img-20220903021308.png#pic_center%20=400x)
然而，以上的方式对于 DNAS 来说有一个缺点，便是搜索时间不弹性，透过上面的图，我们可以发现，当今天我们更新完整个 architecture parameters 之后，我们仅仅只能从 architecture parameters 中 sample 出一组在特定硬体资源限制之下的 neural architecture，因此当今天我们需要搜索在 N 个不同硬体资源限制之下的 neural architectures 时，便需要重新训练 N 次的 architecture parameters 同时重新训练 N 次的 supernet。

## Single-path NAS
而相对于 DNAS 的另外一个种类 single-path NAS，便是直接将整个 NAS 的程序拆成两个独立的步骤，分别为 : Supernet training 以及 Architecture searching。
1、Supernet training：在这个阶段，每次只会从 supernet 当中 sample 出 single path 并且更新其对应到的参数，透过这样的方式，除了可以模拟实际在 search space 当中离散的 neural architecture，同时也可以大幅度的减少 GPU memory 需求。而如何训练 supernet 在现今的 NAS 研究当中也是一个很重要的研究方向 [8][10][11]，原因在于我们会希望 Supernet 是一个可以正确评估 neural architectures 好坏的 performance estimator，因此假如 supernet 并没有被稳定且公平的训练，会使得 supernet 本身产生一定的 bias 进而使得我们没有办法良好的评估 neural architectures 之间的好坏。
2、Architecture searching：当 supernet 训练完毕之后，supernet 本身便可以作为一个 performance estimator，并且结合不同的 search strategy(e.g., random search 以及 evolution algorithm) 来进行搜索。