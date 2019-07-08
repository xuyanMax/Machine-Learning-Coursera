# Machine-Learning
## ML Concepts
### Framing
#### Key ML Terminology 
- Regression & Classification: predict continuous & discrete values
- Models: defines the relationship between features and label
- Labels & Features: a label is the thing we're predicting; a feature is an input variable

### Descending to ML
#### Training & Loss
Training
> learning (determining) good values for all the weights and the bias from labeled examples. In supervised learning, a machine learning algorithm builds a model by examining many examples and attempting to find a model that minimizes loss; this process is called empirical risk minimization.

loss 
> is a number indicating how bad the model's prediction was on a single example

### Reducing Loss
#### Gradient Descent

Gradient Descend
> The gradient always points in the direction of steepest increase in the loss function. The gradient descent algorithm takes a step in the direction of the negative gradient in order to reduce loss as quickly as possible.

#### Learning Rate (Hyperparameters are the knobs programmers tweak in ML)
Learning Rate
: Gradient descent algorithms multiply the gradient by a scalar known as the learning rate (also sometimes called step size) to determine the next point

#### Batch
Batch
> a batch is the total number of examples you use to calculate the gradient in a single iteration

#### Stochastic Gradient Descent (SGD)

Batch Gradient Descent
> 每次使用`全量的训练集`样本来更新模型参数; 批量梯度下降每次学习都使用`整个训练集`，因此其`优点`在于每次更新都会朝着正确的方向进行，最后能够保证收敛于极值点(凸函数收敛于全局极值点，非凸函数可能会收敛于局部极值点)，但是其`缺点`在于每次学习时间过长，并且如果训练集很大以至于需要消耗大量的内存，并且全量梯度下降不能进行在线模型参数更新。

Stochastic Gradient Descent
> `随机梯度下降`算法每次从训练集中`随机选择一个样本`来进行学习，
批量梯度下降算法每次都会使用全部训练样本，因此这些计算是冗余的，因为每次都使用完全相同的样本集。而随机梯度下降算法每次只随机选择一个样本来更新模型参数，因此每次的学习是非常快速的，并且可以进行在线更新。随机梯度下降最大的`缺点`在于每次更新可能并不会按照正确的方向进行，因此可以带来`优化波动(扰动)`，由于波动，因此会使得`迭代次数（学习次数）增多`，即`收敛速度变慢`。不过最终其会和全量梯度下降算法一样，具有相同的收敛性，即凸函数收敛于全局极值点，非凸损失函数收敛于局部极值点。

Mini-Batch SGD
> a compromise between full-batch iteration and SGD. A mini-batch is typically between 10 and 1,000 examples, chosen at random. Mini-batch SGD reduces the amount of noise in SGD but is still more efficient than full-batch. 相对于随机梯度下降，`Mini-batch梯度下降降低了收敛波动性`，即降低了参数更新的方差，使得更新更加稳定。相对于全量梯度下降，其`提高了每次学习的速度`。并且其不用担心内存瓶颈从而可以利用矩阵运算进行高效计算。一般而言每次更新随机选择[50,256]个样本进行学习，但是也要根据具体问题而选择，实践中可以进行多次试验，选择一个更新速度与更次次数都较适合的样本数。`mini-batch梯度下降可以保证收敛性，常用于神经网络中`。


