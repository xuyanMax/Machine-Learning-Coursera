# Machine-Learning
## ML Concepts
### 1. Framing
#### 1.1 Key ML Terminology 
- Regression & Classification: predict continuous & discrete values
- Models: defines the relationship between features and label
- Labels & Features: a label is the thing we're predicting; a feature is an input variable

### 2. Descending to ML
#### 2.1 Training & Loss
Training
> learning (determining) good values for all the weights and the bias from labeled examples. In supervised learning, a machine learning algorithm builds a model by examining many examples and attempting to find a model that minimizes loss; this process is called empirical risk minimization.

loss 
> is a number indicating how bad the model's prediction was on a single example

### 3. Reducing Loss
#### 3.1 Gradient Descent

_Gradient Descent_
> The gradient always points in the direction of steepest increase in the loss function. The gradient descent algorithm takes a step in the direction of the negative gradient in order to reduce loss as quickly as possible.

#### 3.2 Gradient Descent Optimizations
> todo 

#### 3.3 Learning Rate (Hyperparameters are the knobs programmers tweak in ML)
Learning Rate
> Gradient descent algorithms multiply the gradient by a scalar known as the learning rate (also sometimes called step size) to determine the next point

#### 3.4 Batch
Batch
> a batch is the total number of examples you use to calculate the gradient in a single iteration

#### 3.5 Stochastic Gradient Descent (SGD)

Batch Gradient Descent
> 每次使用`全量的训练集`样本来更新模型参数; 批量梯度下降每次学习都使用`整个训练集`，因此其`优点`在于每次更新都会朝着正确的方向进行，最后能够保证收敛于极值点(凸函数收敛于全局极值点，非凸函数可能会收敛于局部极值点)，但是其`缺点`在于每次学习时间过长，并且如果训练集很大以至于需要消耗大量的内存，并且全量梯度下降不能进行在线模型参数更新。

Stochastic Gradient Descent
> `随机梯度下降`算法每次从训练集中`随机选择一个样本`来进行学习，
批量梯度下降算法每次都会使用全部训练样本，因此这些计算是冗余的，因为每次都使用完全相同的样本集。而随机梯度下降算法每次只随机选择一个样本来更新模型参数，因此每次的学习是非常快速的，并且可以进行在线更新。随机梯度下降最大的`缺点`在于每次更新可能并不会按照正确的方向进行，因此可以带来`优化波动(扰动)`，由于波动，因此会使得`迭代次数（学习次数）增多`，即`收敛速度变慢`。不过最终其会和全量梯度下降算法一样，具有相同的收敛性，即凸函数收敛于全局极值点，非凸损失函数收敛于局部极值点。

Mini-Batch SGD
> a compromise between full-batch iteration and SGD. A mini-batch is typically between 10 and 1,000 examples, chosen at random. Mini-batch SGD reduces the amount of noise in SGD but is still more efficient than full-batch. 相对于随机梯度下降，`Mini-batch梯度下降降低了收敛波动性`，即降低了参数更新的方差，使得更新更加稳定。相对于全量梯度下降，其`提高了每次学习的速度`。并且其不用担心内存瓶颈从而可以利用矩阵运算进行高效计算。一般而言每次更新随机选择[50,256]个样本进行学习，但是也要根据具体问题而选择，实践中可以进行多次试验，选择一个更新速度与更次次数都较适合的样本数。`mini-batch梯度下降可以保证收敛性，常用于神经网络中`。 
 
### 4. Training & Test Sets

Most of time, we split the data into training set and testing set, with a ratio of 8:2. With two partitions, the workflow could look as: 
1. train model on `training set`
2. evaluate model on `test set`
3. `tweak model` according to results on test set, and go to step 1 to retrain a new model 

However good it is, training and test sets partition is not a panacea. Dividing your dataset into three subsets can greatly reduce your possibilities of overfitting. The possible workflow is like this:
1. train model on `training set`
2. evaluate model on `validation set`
3. `tweak model` according to results on validation set and go to step 1 to retrain 
4. after training several models, `pick a model` does best on validation set and confirm/double-check results on `test set`

**Training Set** -  a subset to train a model

**Test Set** - a subset to test the trained model

- Large enough to yield statistically meaningful results 
- Representative of the data set as a whole. In other words, never pick a test set with different characteristics/features than the training set

### 5. Representations
#### 5.1 Feature Engineering
>transform raw data into a feature vector

**Mapping numeric values** 

Integer and floating-point data don't need a special encoding. Mapping Integer values to Float values is unnecessary.

**Mapping categorical values**

Like sign language and rock-paper-scissor datasets, they are categorical features, which requires a discrete set of possible values. We can directly map these categories (type of String) into numbers (0,1,2,3...). But there will be some constraints. It is better instead to create a binary vector (the vector length iss equal to the number of elements in the categorical feature)for each categorical feature in the model:
- for values that apply to the example, set corresponding vector element(s) to 1
- set all other elements to 0

**Constraints?**

**One-hot encoding** - a single value is 1 in a vector

**Multi-hot encoding** - multiple values are 1 in a vector

#### Qualities of Good Features

**What makes good features?**
- Avoid rarely used discrete feature values (appearance in the dataset <= 5).
- Clear and obvious meanings 
- No `magic` values (accentricity, out of actual range) with actual data. 

**Deal with accentric values**

- For discrete variables, add a new value to the set and use it to signify that the feature value is missing. For example, create a Boolean feature that indicates whether or not a this feature was supplied.
- For continuous variables, ensure missing values don't affect the model by using the `mean` value of the feature's data.

#### 6. Cleaning Data

##### 6.1. SCALING FEATURE VALUES
> convert feature values into a `standard range` (like [-1, 1], [0, 1]). 
 
Feature Scaling is no point for linear regression where there is only a single feature, but it provides a lot benefits for multiple features problem: 
- help gradient descent converge more quickly.
- help avoid the `NaN trap`, in which one number can overflows the precision.
- help the model learn appropriate weight for each feature. Without feature scaling, the model will pay much more attention to the features having a wide range.

**Handle extreme outliers**

For a feature distribution having a lonnng tail, we could take the log of every value `log(val + 1)`. Log scaling usually does a better job. 

Sometimes, taking log operations still leaves a significant tail of outlier values. We need `clip` the maximum value of the feature at an arbitrary value, say X. In other words, we don't ignore those outliers but make those greater than X to be X. 

##### 6.2. Bining 

Features like `hourse_latitude`, it doesn't make sense to represent `latitude` as a floating point feature in the model in that there is no linear relationship between lattitude and housing values. In addition, individual latitudes probably are not a good indicator of house values. 

Therefore, we divide latitude into `bins` or `buckets` such as `latitudeBin1`, `latitudeBin2`, ..., `latitudeBin11`. Instead of having 11 separate features, it is better to unite them into a single 11-element vector.

`[0, 0, 0, 0, 1, 0, 0, 0, ,0, 0, 0]`

```python
def select_and_transform_features(source_df):
  LATITUDE_RANGES = zip(range(32, 44), range(33, 45))
  selected_examples = pd.DataFrame()
  selected_examples["median_income"] = source_df["median_income"]
  for r in LATITUDE_RANGES:
    selected_examples["latitude_%d_to_%d" % r] = source_df["latitude"].apply(
      lambda l: 1.0 if l >= r[0] and l < r[1] else 0.0)
  return selected_examples

```

##### 6.3. Scrubbing 

### 7. Feature Crosses
 > A feature cross is a synthetic feature that encodes nonlinearity in the feature space by multiplying two or more input features together. (The term cross comes from cross product.) Let's create a feature cross named x3 by crossing x1 and x2: x3 = x1*x2
 
#### 7.1 Kinds of feature crosses
 - [x1*x2]: cross product of two features 
 - [x1*x2*x3*x4...*xn]: cross product of multiple features
 - [x1*x1]: square of a feature

#### 7.2 Crossing One-Hot Vectors
Usually, cross product seldom happens to models with continuous features, but it does frequently happens to one-hot feature vectors. You can think of feature cross of one-hot featur vectors as `logical conjunction`.  For example, you bin `latitude` and `longitude`, generating one-hot five-element feature vectors. 

`binned_latitude=[0, 0, 1, 0, 0]`

`binned_longitude=[0, 1, 0, 0, 0]`

The cross feature will be a 25-element on-hot vector(24 zeros and 1 one), in which the single one identifies a particular conjunction of latitude and longitude, which may be a good indicator of house values.

### 8. Regularization: simplicity 

**Overfitting** 
> If we use a model that is too complicated, such as one with too many crosses, we give it the opportunity to fit to the noise in the training data, often at the cost of making the model perform badly on test data.
 
 **Regularization**
 > A way to prevent overfitting by penalizing complex models, a principle called regularization.
 
#### 8.1. L2 Regularization
we could prevent overfitting by `penalizing complex models`, a principle called `regularization`.

In other words, instead of simply aiming to minimize loss (empirical risk minimization):

we'll now minimize `loss+complexity`, which is called `structural risk minimization`:

Our training optimization algorithm is now a function of two terms: the loss term, which measures how well the model `fits the data`, and the regularization term, which measures `model complexity`.

If model complexity is a `function of weights`, a feature weight with a high absolute value is more complex than a feature weight with a low absolute value.

We can quantify complexity using the L2 regularization formula, which defines the regularization term as the `sum of the squares` of all the feature weights

In this formula, weights close to zero have little effect on model complexity, while outlier weights can have a huge impact.

#### 8.2. Lambda 
Model developers tune the overall impact of the regularization term by multiplying its value by a scalar known as `lambda` (also called the `regularization rate`). That is, model developers aim to do the following:

`minimize(Loss(Data|Model|) + LAMBDA*complexity(Model))`

When choosing a lambda value, the goal is to strike the right balance between simplicity and training-data fit:
    - If your lambda value is `too high`, your model will be `simple`, but you run the risk of `underfitting` your data. Your model won't learn enough about the training data to make useful predictions.
    - If your lambda value is `too low`, your model will be more `complex`, and you run the risk of `overfitting` your data. Your model will learn too much about the particularities of the training data, and won't be able to generalize to new data

### 9. Logistic Regression
#### 9.1. Calculating a Probability
#### 9.2. Loss and Regularization

### 10. Classification 
#### 10.1. Thresholding
#### 10.2. Accuracy
#### 10.3. Precision and Recall
#### 10.4. ROC Curve and AUC
#### 10.5. Prediction Bias

### 11. Regularization: sparsity 
#### 11.1. L1 Regularization

### 12. Neural networks
#### 12.1. Structure 

### 13. Train Neural networks
#### 13.1 Best Practices

### 14. Multi-Class Neural Nets
#### 14.1. One vs. All
#### 14.2. Softman

### 15. Embeddings 
#### 15.1. Motivation from Collaborative Filtering
#### 15.2. Categorical Input Data
#### 15.3. Translating to a Lower Dimensional space
#### 15.4. Obtaining Embeddings



