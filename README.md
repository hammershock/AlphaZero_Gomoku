## AlphaZero-Gomoku
这是一个基于 AlphaZero 算法实现的简单棋盘游戏五子棋（又名 Gobang 或 Five in a Row）的自我对弈训练。五子棋相较于围棋或国际象棋要简单得多，这使我们能够专注于 AlphaZero 的训练方案，并在几小时内就在单台个人电脑上训练出一个相当不错的AI模型。

参考资料：
1. AlphaZero：通过自我对弈使用通用强化学习算法掌握国际象棋和将棋
2. AlphaGo Zero：无需人类知识，掌握围棋游戏

### 2018年2月24日更新：支持使用 TensorFlow 训练!
### 2018年1月17日更新：支持使用 PyTorch 训练!

### 训练模型之间的示例游戏
- 每一步使用400次MCTS模拟：  
![playout400](https://raw.githubusercontent.com/junxiaosong/AlphaZero_Gomoku/master/playout400.gif)

### 系统要求
想要与训练好的AI模型对弈，需要：
- Python >= 2.7
- Numpy >= 1.11

想要从头开始训练AI模型，还需要以下之一：
- Theano >= 0.7 和 Lasagne >= 0.1      
或者
- PyTorch >= 0.2.0    
或者
- TensorFlow

**注意**：如果你的 Theano 版本 > 0.7，请按照这个 [问题](https://github.com/aigamedev/scikit-neuralnetwork/issues/235) 安装 Lasagne，
否则，请使用 pip 将 Theano 降级至 0.7 ``pip install --upgrade theano==0.7.0``

如果你想使用其他的深度学习框架训练模型，你只需要重写 policy_value_net.py 文件。

### 入门
想要使用提供的模型进行对弈，请从目录运行以下脚本：
```
python human_play.py  
```
你可以修改 human_play.py 来尝试不同的提供模型或纯 MCTS。

想要从头开始训练AI模型，使用 Theano 和 Lasagne，直接运行：
```
python train.py
```
使用 PyTorch 或 TensorFlow，首先需要修改文件 [train.py](https://github.com/junxiaosong/AlphaZero_Gomoku/blob/master/train.py)，即注释掉一行
```
from policy_value_net import PolicyValueNet  # Theano and Lasagne
```
并取消注释行
```
# from policy_value_net_pytorch import PolicyValueNet  # Pytorch
或
# from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
```
然后执行：``python train.py`` （若要在 PyTorch 中使用 GPU，请设置 ``use_gpu=True`` 并在 policy_value_net_pytorch.py 中的函数 train_step 使用 ``return loss.item(), entropy.item()``，如果你的 PyTorch 版本大于 0.5）

模型（best_policy.model 和 current_policy.model）将在每几次更新后保存（默认50次）。

**注意**：提供的4个模型是使用 Theano/Lasagne 训练的，要在 PyTorch 中使用它们，请参考 [问题 5](https://github.com/junxiaosong/AlphaZero_Gomoku/issues/5)。

**训练提示**：
1. 建议从6 * 6的棋盘和4连珠开始。在这种情况下，我们可能在大约2小时内通过500~1000场自我对弈获得一个合理的模型。
2. 对于8 * 8棋盘和5连珠的情况，可能需要2000~3000场自我对弈来获得一个好的模型，这可能需要在单个PC上花费大约2天的时间。

### 深入阅读
我的文章描述了

一些实现的细节，中文版可在这里查看：[https://zhuanlan.zhihu.com/p/32089487](https://zhuanlan.zhihu.com/p/32089487)