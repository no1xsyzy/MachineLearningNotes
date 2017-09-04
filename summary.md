#前言
全篇为Andrew Ng（吴恩达）的机器学习课程的笔记，亦作机器学习方面的调研报告。

* 虽说作为调研报告，只因以笔记方式写就，其中不免充满碎片化的语言和思想；
* 机器学习作为计算机科学的内容，自然免不了诸多公式，虽然尽力作了解释，但终究难以写下整个推倒过程，自然会有难以通顺之处；
* 本来这课程需要线性代数和概率论基础作为前置课程，如果说要在基础方面写得面面俱到，未免过于冗长，鄙人口齿又不伶俐，故而假定读者拥有相关知识，抑或是能够查找相关资料。

综上几方面还请读者多多海涵。

为表敬意，给出几处原课程的来源，亦供参考：

* [Coursera <https://zh.coursera.org/learn/machine-learning>](https://zh.coursera.org/learn/machine-learning)
* [YouTube <https://www.youtube.com/playlist?list=PLA89DCFA6ADACE599>](https://www.youtube.com/playlist?list=PLA89DCFA6ADACE599)
* [网易公开课 <http://open.163.com/special/opencourse/machinelearning.html>](http://open.163.com/special/opencourse/machinelearning.html)

另外，还有原课程主页：

* [CS229 <http://cs229.stanford.edu/>](http://cs229.stanford.edu/)

#机器学习的动机与应用
##何为机器学习
不通过特定的逻辑而是通过让计算机学习来处理事务。

或者

对于经历$E$、任务$T$和评价标准$P$，通过对$E$的分析，增强计算机在$T$上的$P$。

##监督学习（supervised learning）
学习存在的$X^* \rightarrow Y$，通过$x$求出$y$。类似于回归，但能处理更复杂的问题。

##非监督学习（unsupervised learning）
并不是先给出正解，比如集群（cluster）

##增强学习（reinforcement learning）
针对需要进行一系列决策，并且不需要全部的决策是好的，只要保证有足够多的好的决策，失误并不会造成整体问题。类似闭环控制，扰动并不会导致最终结果误差过大。

#监督学习-梯度下降
本节的内容主要是梯度下降（gradient descent）算法的原理。

##符号
* $m$：训练样本大小
* $x$：输入变量/特征
* $y$：输出变量/目标变量
* $\left(x^{\left(i\right)} , y^{\left(i\right)}\right)$：第(i)组训练样本
* $\left(x,y\right)$：训练样本
* $h$：假设，即学习后的结果
* $\theta_{\left(n+1\right)\times 1}$：训练参数

##批量梯度下降法（batch gradient descent/BGD）
实质就是爬山算法：初始设定theta为0，通过向方差更低的位置不断移动，能够找出局部最优点。
移动的过程通过计算偏微分来向较低处移动。

##随机梯度下降法（stochastic gradient descent/SGD）
每次仅对一个样本学习，最终也可以收敛于最优解。

##正规方程（normal equations）
通过很长一段推导，最终将目标<方差最小化>转化为一个方程$$X^{T}X\theta=X^{T}y$$

#欠拟合与过拟合的概念
##欠拟合（underfitting）与过拟合（overfitting）

拟合的假设（hypothesis）有着过少的参数，导致其中很多性质不能被提取，称为欠拟合（underfitting）

相反地，拟合的假设有着过多的参数，导致尝试符合每一个点，导致无法得出趋势，称为过拟合（overfitting）

为了不发生这种情况，提出了局部加权回归。

#局部加权回归（Locally~weighted~regression/loess/lowess）
局部加权回归算法的主要思想是：在每一处需要得到目标值的位置，靠近位置的点得到更多权重，而每次回归一条直线。这样的算法能够对各种情况进行处理，而不需要预先考虑总的函数式。

#分类算法（Classification）
分类算法主要考虑目标变量始终为0或者1的情况。仅有两种情况的时候线性回归很多时候并不能工作，这时候提出采用另外的函数表示取1的概率。推导出的公式和线性回归的公式非常相似但是其中函数改变（变为非线性）导致实际情况不同。

#牛顿方法
##牛顿方法（Newton's Method）
牛顿方法，学校那边用的名字是Newton-Raphson method，其实是一样的。采用更高阶下的牛顿方法，寻找似然性最大的$\theta$的过程可以变得非常快。

最终的结果是：
$$\theta^{\left(t+1\right)}=\theta^{\left(t\right)-H^{-1}\nabla_{\theta}l}$$
其中$H$称为海森矩阵（Hessian matrix），并且$H_{ij}=\frac{\partial^2 l}{\partial \theta_i\,\partial \theta_j}$

#指数分布族（Exponential family）
可以知道，高斯分布、伯努利分布和多项式分布都属于指数分布族
$$ P(y;\eta) = b(y)\exp(\eta^T T(y)-a(\eta)) $$
通过给定a,b,T能够得到高斯分布、伯努利分布和多项式分布。

##伯努利分布
$$
\begin{aligned}
a(\eta)&=-\log(1-\phi)=\log(1+e^\eta)
b(y)&=1
T(y)&=y
\end{aligned}
$$

##高斯分布
$$
\begin{aligned}
a(\eta)&=\frac{1}{2}\eta^2
b(y)&=\frac{1}{\sqrt(2\pi)}\exp(-\frac{1}{2}y^2)
T(y)&=y
\end{aligned}
$$

#广义线性模型（Generalized linear model/GLM）
可以将各种模型代入广义线性模型。

将多项式分布一顿推导得到了Softmax回归。似然度的对数是个二重求和。

#生成学习算法
##生成学习算法（Generative Learning algorithms）
与之前的算法不同，生成学习算法尝试生成分类目标的特征模型，然后测试新数据与模型的吻合程度。
根据贝叶斯法则，
$$p\left(y|x\right)=\frac{p\left(y|x\right)p\left(y\right)}{p\left(x\right)}$$

##高斯判别分析（Gaussian discriminant analysis）
在很多情况下特征符合或者近似符合多维高斯分布，所以可以简单地使用推导结果[^1]

[^1]: <http://cs229.stanford.edu/notes/cs229-notes2.pdf> Page 6

##朴素贝叶斯（Naive Bayes）
* 问题：垃圾邮件分类
* 建模：假定有一组n个单词。每一封邮件的特征提取为“是否存在这一个单词”的n维列向量。
* 假设：知道是否为垃圾邮件的情况下，单词的存在于否之间不存在相关性。
* 推倒得出：
$$
\begin{aligned}
p\left( y=1|x \right) &= \frac{ p\left( x|y=1 \right) p\left( y=1 \right) }{ p\left( x \right) }\\
&=\frac{ \left( \prod^n_{i=1}p\left( x_i|y=1 \right) \right) p\left( y=1 \right) }{ \left( \prod^n_{i=1}p\left( x_i|y=1 \right) \right) p\left( y=1 \right) + \left( \prod^n_{i=1}p\left( x_i|y=0 \right) \right) p\left( y=0 \right) }
\end{aligned}
$$

##拉普拉斯平滑（Laplace smoothing）
比如，统计上得到了5次零，那么第6次是零的概率是多少？

* 经典的处理方式是计算频率作为概率：$\phi_j=\frac{\sum^m_{i=1}1\left\{ z^{\left(i\right)} =j \right\} }{m}$
    - 这种处理方式的问题是：如果不发生，即 $\forall i\in\left\{1,2,\ldots,m\right\}, z^{\left(i\right)} \neq j$
    - 频率是零，导致计算得到的概率也是零，那么接下来这件事情不会发生。这样的计算方式显然是不合理的。
* 拉普拉斯平滑，将每种情况的计数增加一：$\phi_j=\frac{\sum^m_{i=1}1\left\{ z^{\left(i\right)} =j \right\}+1 }{m+k}$
    - 即使不发生的情况也有概率，同时又保证了各个性质不变。

#朴素贝叶斯算法
##新的垃圾邮件过滤器模型
对每封邮件建模，使得一个单词变为一个整数，表示该单词为单词表中的第几个，然后对是否是垃圾邮件的每种情况分析每个位置出现每个单词的概率。其中也用到拉普拉斯平滑。

#支持向量机（support vector machine/SVM）
若预测恰为$1\left\{\theta^T x \geq 0\right\}$，
反过来说只要得到一个能够实现这一分隔的$\theta$就可以了。

##记号
$$y\in \left\{-1,+1\right\}$$

$$
g\left(z\right)=
\begin{cases}
  1, &\text{if } z\geq 0\\
  -1, &\text{otherwise}
\end{cases}
$$

$$h_{w,b}\left(x\right)=g(w^T x+b)$$

##几何间隔
$$\gamma^{\left(i\right)} = y^{\left(i\right)} \left( \left( \frac{w}{\|w\|} \right)^T x^{\left(i\right)} + \frac{b}{\|w\|} \right)$$

#最优间隔分类器问题
主要为对支持向量机证明和运算简化的讲解。（？）

虽然每一处单独看都能看明白但是整体来看还是太复杂。

#顺序最小优化算法
##核函数（Kernel）
为了能够表示复杂的特征，采用核函数将两个原始特征向量以一个复杂度较低的函数映射到高维空间内的内积。

$$
\begin{aligned}
&K\left(x^{\left(i\right)},x^{\left(j\right)}\right)=\left \langle \phi\left(x^{\left(i\right)}\right),\phi\left(x^{\left(j\right)}\right) \right \rangle\\
&K : \mathbb{R}^n \times \mathbb{R}^n \mapsto \mathbb{R}
\end{aligned}
$$
一个例子：
$$
\begin{aligned}
K\left ( x,z \right )&=\left ( x^T z \right )^2\\
&=\left ( \sum_{i=1}^{n}x_i z_i \right )\left ( \sum_{j=1}^{n}x_j z_j \right )\\
&=\sum_{i=1}^{n}\sum_{j=1}^{n}\left ( x_i x_j \right )\left ( z_i z_j \right )\\
\phi\left ( x \right )&=\begin{bmatrix}
x_1x_1\\
x_1x_2\\
x_1x_3\\
x_2x_1\\
x_2x_2\\
x_2x_3\\
x_3x_1\\
x_3x_2\\
x_3x_3
\end{bmatrix}
\end{aligned}
$$

将3维的特征以$\phi$的方式映射到9维，并且运算开支非常小。甚至一些核函数可以将特征映射到无穷维空间，这是通过常规方式无法实现的：计算机不能处理无穷的数据。

核函数是否是可用的（valid）可以通过这样判断：
对于$K : \mathbb{R}^n \times \mathbb{R}^n \mapsto \mathbb{R}$，

##软间隔（soft margin）
有的时候，通过设定核函数并不能轻易地将数据组分开，比如有一个/些异常数据，那么尝试将这个数据正确分类反而可能造成更多数据预测分类不正确。

所以，引入软间隔，使得一些点可以不正确分类，同时对这样的数加入惩罚量。问题变为：
$$
\begin{aligned}
\min_{\gamma,w,b}\quad &\frac{1}{2}\left \| w \right \|^2+C\sum_{i=1}^{n}\xi_1\\
\text{s.t.}\quad &y^{\left( i \right )}\left(w^T x^{\left(i \right )} +b \right ) \geq 1-\xi_i,\quad i=1,\dots ,m\\
&\xi_i \geq 0,\quad i=1,\dots,m
\end{aligned}
$$
同样是凸优化问题。

##SMO算法
###坐标下降法
在凸优化问题上，可以每次找一个变量的最优，不断轮换找最优的变量能够得到最终的全局最优。

###SMO算法
因为$a_i$之间具有关联性，如果$a_2,\dots,a_m$固定，那么$a_1$也必须固定，所以转而采用两个一起改变的策略。每次改变两个变量，找到其中的最优解。

#学习理论
##偏差/方差权衡
如第一讲中所述，

*如果设定假设过于简单，大多数的训练样本不在拟合曲线上，数据误差较大，称欠拟合。
*如果设定假设过于复杂，无法表现变化趋势，称过拟合。

这两者均有很大的泛化误差（generalization error）。

* 误差偏差意味着，在训练集趋向于无穷大的时候依旧会出现的误差。
* 误差方差是指仅在数据量很小的情况下会产生的误差。

###引理1
使得$A_1,A_2,\dots,A_k$是$k$个不同的事件（相关性不作假设），有
$$P\left( A_1 \cup \dots \cup A_k \right) \leq P\left( A_1 \right)+\dots+P\left( A_k \right)$$

###引理2
使得$Z_1,\dots,Z_m$为$m$个独立恒等分布随机变量，
且遵循伯努利$\mathrm{Bernoulli}\left( \phi \right)$分布。
即$P\left( Z_i=1 \right)=\phi$且$P\left( Z_i=1 \right)=\phi$。
使得$\hat{\phi}=\frac{1}{m}\sum_{i=1}^{m}Z_i$为所有随机变量的平均数，对于任意确定的$\gamma>0$，有
$$P\left( \left|\phi-\hat{\phi} \right|>\gamma \right)\leq 2 \exp( -2\gamma^2 m)$$
这意味着$\hat{\phi}$与$\phi$相差较远的可能性很低。

可以证得所有的假设的泛化误差有
$$\varepsilon(\hat{h})\leq\left(\min_{h\in\mathcal{H}}\varepsilon \left ( h \right ) \right )+2\sqrt{\frac{1}{2m}\log \frac{2k}{\delta }}$$

若$\left|\mathcal{H}\right|=k$且$\delta$与$\gamma$确定，对于$\varepsilon(\hat{h})\leq \min_{h\in\mathcal{H}}\varepsilon(h)+2\gamma$具有至少$1-\delta$的概率的情况下，
$$
\begin{aligned}
m&\geq \frac{1}{2\gamma^2}\log \frac{2k}{\delta}\\
&=O\left( \frac{1}{\gamma^2}\log \frac{k}{\delta} \right)
\end{aligned}
$$

当假设的数量变多的时候，$\left(\min_{h\in\mathcal{H}}\varepsilon\left(h\right) \right)$变小，同时，$2\sqrt{\frac{1}{2m}\log\frac{2k}{\delta}}$变大。前者通常被视作偏差，后者通常被视作方差。

#学习理论
##$\mathcal{H}$为无穷集的情况
###引入VC维度
给定集合$\mathcal{S}=\left\{ x^{\left( 1 \right)},\dots,x^{\left( d \right)} \right\}$为一点集（与训练集无关）$x^{^{\left( i \right)}}\in\mathcal{X}$，若$\mathcal{H}$能够实现对$\mathcal{S}$的任意划分，称$\mathcal{H}$**打散**了$\mathcal{S}$。

而对于给定的假设集$\mathcal{H}$，其VC维$\mathrm{VC} \left( \mathcal{H} \right)$是最大的能打散的点集的大小。

###定理
给定$\mathcal{H}$，使得$d=\mathrm{VC} \left ( \mathcal{H} \right )$，在至少$1-\delta$的可能性下，对于所有$h\in \mathcal{H}$，
$$\left| \varepsilon\left( h \right)-\hat{\varepsilon}\left( h \right) \right| \leq O \left( \sqrt{ \frac{d}{m} \log\frac{m}{d} + \frac{1}{m} \log\frac{1}{\delta} } \right)$$
因此也有，在至少$1-\delta$的可能性下，
$$\varepsilon\left( \hat{h} \right) \leq \varepsilon\left( h^* \right)+O \left( \sqrt{ \frac{d}{m} \log\frac{m}{d} + \frac{1}{m} \log\frac{1}{\delta} } \right)$$

###结论
练习样本的数量和训练参数的数量成正比。

#模型选择
有一组有限多的模型，$\mathcal{M}=\left\{ M_1,\dots,M_d \right\}$
要选取一个最好的模型

##交叉验证
假设这种方法：

1. 对每个模型训练出一个假设
2. 选择训练误差最小的一个

显然是不行的，因为过拟合的训练误差最小。

###简单交叉验证

1. 随机选取70%的数据作为训练数据$\mathcal{S}_{\textrm{train}}$，剩下的作为交叉验证数据集$\mathcal{S}_{\textrm{CV}}$。
2. 使用每个模型$M_i$，仅对$\mathcal{S}_{\textrm{train}}$进行训练，得到假设$h_i$。
3. 对每个假设$h_i$计算在交叉验证数据集$\mathcal{S}_{\textrm{CV}}$上的经验误差$\hat{\varepsilon}_{\mathcal{S}_{\textrm{CV}}}\left( h_i \right)$。选择经验误差最小的。

###切分交叉验证

1. 将$\mathcal{S}$随机切分为$k$个互不重叠的训练样本$\mathcal{S}_1,\dots,\mathcal{S}_k$
2. 对每个模型$M_i$和每个训练样本切片$\mathcal{S}_j$，训练整个样本除了这个切片之外的所有切片的并集，得到假设$h_{ij}$，计算在$\mathcal{S}$上的经验误差$\hat{\varepsilon}_{\mathcal{S}_j\left( h_{ij} \right)}$。
3. 选择经验误差平均值最小的模型$M_i$，然后对整个训练样本训练一个假设作为最终假设。

通常$k=10$，其他数值也可以。特别地，$k=m$也就是每次只排除一个，称为**排一交叉验证**。

##特征选取
###正向搜索
初始特征集$\mathcal{F}=\varnothing$，不断尝试增加一个特征，增加特征的过程会扫描整个特征集，选择交叉验证的结果最好的。最终选择最好的特征子集。

###反向搜索
与正向搜索相反，初始特征集$\mathcal{F}$为特征全集。而每次尝试删除一个特征。

###筛选特征选择
测试每个特征的信息量。（应该是类似香农的信息论的观点）

#模型选择
##贝叶斯统计和正规化
与之前的频率学派观点不同，贝叶斯学派的观点将参数视作随机变量，而似然性变为参数符合推论的可能性。而特征选择的过程可以参考参数的后验分布，通过最大后验估计知道参数符合以0为中心的正态分布。也就是说大多数的参数很小。

##动态学习
有些情况下会需要随着预测同时学习，采用动态学习方法。
<http://cs229.stanford.edu/notes/cs229-notes6.pdf>

#实际经验

##高预测误差
观察训练误差
* 训练误差很小 - 高方差
* 训练误差也大 - 高偏差

- 更多训练样本 - 解决高方差
- 更少训练特征 - 解决高方差
- 更多训练特征 - 解决高偏差
- 尝试其他种类特征 - 解决高偏差
- 更多次梯度下降 - 解决优化算法
- 尝试牛顿法 - 解决优化算法
- 尝试其他的$\lambda$值 - 解决优化目标
- 尝试使用SVM - 解决优化目标

##其他常见问题
* 函数是否在收敛？
* 是否在优化正确的函数？
* 加权？
* $\lambda$的值是否正确？

#非监督学习-集群问题
给定训练集$\left\{ x^{\left( 1 \right)},\dots,x^{\left( m \right)} \right\}$，要分割为多个**集群**。值得注意的是，此处只有特征$x^{\left( i \right)} \in \mathbb{R}^n$，但是却没有标签$y^{\left( i \right)}$。

##$k$-平均聚类

1. 随机初始化集群中心为$\mu_1,\mu_2,\dots,\mu_k \in \mathbb{R}^n$。
2. 不断重复以下过程直到收敛：

    a. 对每个$i$，$$c^{\left( i \right)} := \arg \min_j \left\| x^{\left( i \right)}-\mu_j \right\|^2$$
    b. 对每个$j$，$$\mu_j := \frac{\sum_{i=1}^{m}1\left\{ c^{\left( i \right)}=j \right\}x^{\left( i \right)}}{\sum_{i=1}^{m}1\left\{ c^{\left( i \right)}=j \right\}}$$

实际为**失真函数**上的坐标下降法，失真函数为：
$$J\left( c,\mu \right) = \sum_{i=1}^{m}\left\| x^{\left( i \right)}-\mu_{c^{\left( i \right)}} \right\|^2$$

但失真函数$J$不一定是个凸函数，所以$J$坐标下降的结果不一定是最优。

##混合高斯模型
想象这些集群为一系列没有标签的高斯模型的叠加，反过来尝试添加标签。

###混合高斯模型-EM算法（EM,Expect-Maximum，期望-最大化）
不断重复以下过程直到收敛：

1. （E-期望化步骤）对于每个$i,j$，
$$
\begin{aligned}
w_j^{\left( i \right)} &:= p\left( z^{\left( i \right)}=j|x^{\left( i \right)};\phi,\mu,\Sigma \right)\\
&\;=\frac{p\left( x^{\left( i \right)}|z^{\left( i \right)}=j;\mu,\Sigma \right)p\left( z^{\left( i \right)}=j;\phi  \right)}{\sum_{l=1}^{k}p\left( x^{\left( i \right)}|z^{\left( i \right)}=l;\mu,\Sigma \right)p\left( z^{\left( i \right)}=l;\phi  \right)}
\end{aligned}
$$
2. （M-最大化步骤）对于每个$j$，
$$
\begin{aligned}
\phi_j&:=\frac{1}{m}\sum_{i=1}^{m}w_j^{\left( i \right)}\\
\mu_j&:=\frac{\sum_{i=1}^{m}w_j^{\left( i \right)}x^{\left( i \right)}}{\sum_{i=1}^{m}w_j^{\left( i \right)}}\\
\Sigma_j&:=\frac{\sum_{i=1}^{m}w_j^{\left( i \right)}\left( x^{\left( i \right)}-\mu_j \right)\left( x^{\left( i \right)}-\mu_j \right)^T}{\sum_{i=1}^{m}w_j^{\left( i \right)}}
\end{aligned}
$$

其中，$p\left( x^{\left( i \right)}\middle|z^{\left( i \right)}=j;\mu,\Sigma \right)$是以$\mu_j$为均值、$\Sigma_j$为协方差的高斯分布在点$x^{\left( i \right)}$处的密度。

#EM算法
##琴生不等式
对于凸函数$f\left( x \right)$，$$E\left[ f\left( x \right) \right]\geq f\left( EX \right)$$
并且如果$f$为一*严格*凸函数，那么，$E\left[ f\left( x \right) \right]\geq f\left( EX \right)$等价于$X=E\left[ X \right]$。

##EM算法
给定训练集$\left\{ x^{\left( 1 \right)},\dots,x^{\left( m \right)} \right\}$，要求调节一个模型$p\left(x,z\right)$以适合此训练集。

E-期望化步骤：对所有$i$，$$Q_i\left( z^{\left ( i \right )} \right) := p\left ( z^{\left ( i \right )}\middle|x^{\left ( i \right )};\theta \right )$$

M-最大化步骤：$$\theta := \arg \max_\theta\sum_i\sum_{z^{\left ( i \right )}}Q_i\left ( z^{\left ( i \right )} \right )\log \frac{p\left ( x^{\left ( i \right )},z^{\left ( i \right )};\theta \right )}{Q_i\left ( z^{\left ( i \right )} \right )}$$
#EM算法
##EM算法

如果定义
$$
J\left( Q,\theta \right) = \sum_{i}\sum_{z^{\left( i \right)}}Q_i\left( z^{\left( i \right)} \right)\log\frac{p\left( x^{\left( i \right)},z^{\left( i \right)};\theta \right)}{Q_i\left( z^{\left( i \right)} \right)}
$$
则EM算法其实是$J$上的坐标上升，其中：

* **E-期望化步骤**实为对$Q$做坐标上升
* **M-最大化步骤**实为对$\theta$做坐标上升

##重新讨论混合高斯模型
**E-期望化步骤**比较简单，计算：
$$w_j^{\left( i \right)} = Q_i\left( z^{\left( i \right)}=j \right)=P\left( z^{\left( i \right)}=j|x^{\left( i \right)};\phi,\mu,\Sigma \right)$$
**M-最大化步骤**需要通过调节参数达到最大化$J$。
因为$x^{\left( i \right)}|z^{\left( i \right)}=j;\mu,\Sigma \sim \mathcal{N}\left( \mu_j,\Sigma_j \right)$且$p\left( z^{\left( i \right)}=j;\phi \right) = \phi_j$，
$$
\begin{aligned}
J\left( Q,\theta \right) &= \sum_{i=1}^{m}\sum_{z^{\left( i \right)}}Q_i\left( z^{\left( i \right)} \right)\log\frac{p\left( x^{\left( i \right)},z^{\left( i \right)};\phi,\mu,\Sigma \right)}{Q_i\left( z^{\left( i \right)} \right)}\\
&=\sum_{i=1}^{m}\sum_{j=1}^{k}Q_i\left( z^{\left( i \right)}=j \right)\log\frac{p\left( x^{\left( i \right)}|z^{\left( i \right)}=j;\mu,\Sigma \right)p\left( z^{\left( i \right)}=j;\phi \right)}{Q_i\left( z^{\left( i \right)}=j \right)}\\
&=\sum_{i=1}^{m}\sum_{j=1}^{k}w_j^{\left( i \right)}\log\frac{\frac{1}{\left( 2\pi \right)^{n/2}\left| \Sigma_j \right|^{1/2}}\exp\left( -\frac{1}{2}\left( x^{\left( i \right)}-\mu_j \right)^T\Sigma_j^{-1}\left( x^{\left( i \right)}-\mu_j \right) \right)\cdot\phi_j}{w_j^{\left( i \right)}}
\end{aligned}
$$
对$\mu_l$求梯度得到
$$
\nabla_{\mu_l}J\left( Q,\theta \right)=\sum_{i=1}^{m}w_l^{\left( i \right)}\left( \Sigma_l^{-1}x^{\left( i \right)}-\Sigma_l^{-1}\mu_l \right)\mathrel{\overset{\makebox[0pt]{\mbox{\normalfont\tiny\sffamily set}}}{=}}0
$$
解得
$$\mu_l := \frac{\sum_{i=1}^{m}w_l^{\left( i \right)}x^{\left( i \right)}}{\sum_{i=1}^{m}w_l^{\left( i \right)}}$$
同理可解$\Sigma$，对于$\phi$要注意保持$\sum_{j}\phi_j=1$。最终可得
$$\phi_j:=\frac{1}{m}\sum_{i=1}^{m}w_j^{\left( i \right)}$$

##例子-文档分集群
类似之前的垃圾邮件分类，特征建立为$\left\{ x^{\left( 1 \right)},\dots,x^{\left( m \right)} \right\}$，其中$x^{\left( i \right)} \in \left\{ 0,1 \right\}^{n}$，$x_j^{\left( i \right)}$表示在第$\left( i \right)$份文档中是否存在第$j$个单词。

* **E-期望化步骤**：
$$w^{\left( i \right)}=P\left( z^{\left( i \right)}=1|x^{\left( i \right)};\phi_{j|z},\phi \right)$$
* **M-最大化步骤**：
$$
\begin{aligned}
\phi_{j|z=1} &:= \frac{\sum_{i=1}^{m}w^{\left( i \right)}1\left\{ x_j^{\left( i \right)}=1 \right\}}{\sum_{i=1}^{m}w^{\left( i \right)}}\\
\phi_{j|z=0} &:= \frac{\sum_{i=1}^{m}\left( 1-w^{\left( i \right)} \right)1\left\{ x_j^{\left( i \right)}=1 \right\}}{\sum_{i=1}^{m}\left( 1-w^{\left( i \right)} \right)}\\
\phi_z&:=\frac{\sum_{i=1}^{m}w^{\left( i \right)}}{m}
\end{aligned}s
$$

#因子分析
如果待分类的点数不能远大于特征维度，容易导致协方差矩阵不可逆。为了解决这个问题，可对协方差做一些限制。

##对$\Sigma$的限制
$\Sigma$是个对角线矩阵，$$\Sigma_{jj}=\frac{1}{m}\sum_{i=1}^{m}\left( x_j^{\left( i \right)}-\mu_j \right)^2$$
即，$\Sigma_{jj}$是第$j$个维度的数据的方差。

可以更进一步地限制协方差矩阵，使得对角线上的元素全部相等，即$\Sigma=\sigma^2I$，其中$\sigma^2$是可控的，其最大似然估计为$$\sigma^2 = \frac{1}{mn}\sum_{j=1}^{n}\sum_{i=1}^{m}\left( x_j^{\left( i \right)}-\mu_j \right)^2$$

##高斯分布的边缘分布和条件分布
假设此种表示法：
$$
x = \begin{bmatrix}
x_1\\
x_2
\end{bmatrix}
$$
其中$x_1 \in \mathbb{R}^r$，$x_2 \in \mathbb{R}^s$，$x \in \mathbb{R}^{r+s}$。假定$x\sim\mathcal{N}\left( \mu,\Sigma \right)$，其中：
$$
\mu = \begin{bmatrix}
\mu_1\\
\mu_2
\end{bmatrix}
\quad
\Sigma = \begin{bmatrix}
\Sigma_{11}&\Sigma_{12}\\
\Sigma_{21}&\Sigma_{22}
\end{bmatrix}
$$
其中$\mu_1 \in \mathbb{R}^r$，$\mu_2 \in \mathbb{R}^s$，$\Sigma_{11} \in \mathbb{R}^{r\times r}$，$\Sigma_{12} \in \mathbb{R}^{r\times s}$。注意协方差矩阵为对称矩阵，$\Sigma_{12} = \Sigma_{21}^T$。

可以推得边缘分布：
$$x_1 \sim \mathcal{N}\left( \mu_1,\Sigma_{11} \right)$$
条件分布：
$$
\begin{aligned}
x_1|x_2 &\sim \mathcal{N}\left( \mu_{1|2},\Sigma_{1|2} \right) \\
\mu_{1|2} &= \mu_1+\Sigma_{12}\Sigma_{22}^{-1}\left( x_2-\mu_2 \right)\\
\Sigma_{1|2} &= \Sigma_{11}-\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}
\end{aligned}
$$
#因子分析
##因子分析模型
处理$\left(x,z\right)$上的联合分布。
$$
\begin{aligned}
z&\sim\mathcal{N}\left( 0,I \right)\\
x|z&\sim\mathcal{N}\left( \mu+\Lambda z,\Psi \right)\\
\begin{bmatrix}
z\\
x
\end{bmatrix} &\sim \mathcal{N}\left(
\begin{bmatrix}
\vec{0}\\
\mu
\end{bmatrix},
\begin{bmatrix}
I & \Lambda^T \\
\Lambda & \Lambda\Lambda^T+\Psi
\end{bmatrix}
\right)
\end{aligned}
$$
导致整个似然性公式非常复杂，希望直接处理整个方程非常困难。

##因子分析的期望最大化
<http://cs229.stanford.edu/notes/cs229-notes9.pdf>第6页第4章节。

#主成分分析（PCA）
一些特征间可能存在内在联系，那么两者在一个方向上具有较高的方差，对应正交方向上则具有很低的方差。
希望可以有自动处理的方式。

##预处理
标准化平均值为0和方差为1。

1. 使得$\mu = \frac{1}{m}\sum_{i=1}^{m}x^{\left( i \right)}$
2. 替换$x^{\left( i \right)}$为$x^{\left( i \right)}-\mu$
3. 使得$\sigma_j^2 = \frac{1}{m}\sum_i\left( x_j^{\left( i \right)} \right)^2$
4. 替换$x_j^{\left( i \right)}$为$x_j^{\left( i \right)}/\sigma_j$

##选取投影方向
正确的投影方向上方差大，所以选取的$u$符合：
$$
\begin{aligned}
u &= \arg\max_{u:\left\| u \right\|=1}\frac{1}{m}\sum_{i=1}^{m}\left( {x^{\left( i \right)}}^T u \right)^2\\
\frac{1}{m}\sum_{i=1}^{m}\left( {x^{\left( i \right)}}^T u \right)^2 &= \frac{1}{m}\sum_{i=1}^{m}u^Tx^{\left( i \right)}{x^{\left( i \right)}}^T u\\
 &= u^T\left( \frac{1}{m}\sum_{i=1}^{m}x^{\left( i \right)}{x^{\left( i \right)}}^T \right)u\\
&\Rightarrow u\textup{ 是 }\Sigma=\frac{1}{m}\sum_{i=1}^{m}x^{\left( i \right)}{x^{\left( i \right)}}^T\textup{ 的主特征向量 }
\end{aligned}
$$

如果需要一个$k$维的子空间，选择$\Sigma$的前$k$个特征向量$u_1,\dots,u_k$

##主成分分析作用

* 帮助可视化（人类基本只能理解三维以下）
* 压缩数据量（消除大量无用信息）
* 学习过程（消除无关噪音）
* 异常数据检测（如果一个数据不在主成分分析出的子空间上，很有可能是异常数据）
* 优化匹配距离计算（两个数据的相似性应对在子空间上的投影计算）

#主成分分析-PCA
##算法步骤

1. 数据标准化为零均值和单位方差
2. 计算$\Sigma=\frac{1}{m}\sum_{i=1}^{m}x^{\left( i \right)}{x^{\left( i \right)}}^T$
3. 找到$\Sigma$的前$k$个主特征向量

非常巨大的维度的数据（比如人脸识别，100x100维的图像灰度）导致$\Sigma$维度更加巨大（10000x10000维）

##隐含语义索引
通过之前的“第n个单词是否存在”的建模法，一篇文档的内容可能具有50000维。

相似性函数：
$$
\begin{aligned}
\textup{sim}\left( x^{\left( i \right)},x^{\left( j \right)} \right) &= \cos\left( \theta \right)\\
 &= \frac{{x^{\left( i \right)}}^T x^{\left( j \right)}}{\left\| x^{\left( i \right)} \right\|\left\| x^{\left( j \right)} \right\|}
\end{aligned}
$$
对于study和learn两个单词，被认为是不用的单词
但是通过主成分分析，计算的是投影到子空间的结果，两个单词就能更具有相关性。

##奇异值分解（SVD）
$A_{m\times n} = U_{m\times n}D_{n\times n}V^T_{n\times n}$
$U$的列为$AA^T$的特征向量
$V$的列为$A^TA$的特征向量
只需要用Matlab或者Octave的`svd`命令。

#独立成分分析（ICA）
##基本信息
有数据$s \in \mathbb{R}^n$是由n个独立来源生成的。而能观测的内容为$x=As$。其中$A$被称为混合矩阵。不断地观察可以得到一个数据集$\left\{ x^{\left( i \right)};i=1,\dots,m \right\}$而我们的目标是从$x^{\left( i \right)}$中还原$s^{\left( i \right)}$。

在声音分离的例子中，$s^{\left( i \right)}$是一个$n$维向量，$s^{\left( i \right)}_j$表示的是第$j$个说话者在$i$时刻发出的声音，而$x^{\left( i \right)}_j$表示的是第$j$个话筒在$i$时刻收到的声音。

$W=A^{-1}$为还原矩阵，目标是找到$W$，用于还原$s^{\left( i \right)}=Wx^{\left( i \right)}$。为了方便，将$W$的第$i$行记作$w_i^T$。

##歧义

* 假定一个排列矩阵$P$，每一行和每一列均恰有一个$1$，那么$W$和$PW$是无法区分的。
* $w_i$的大小无法区分，如果$A$变为$2A$而$s^{\left( i \right)}$变为$0.5s^{\left( i \right)}$，那么结果$x^{\left( i \right)}$也是一样的。

但是，对于声音分离的例子，这些歧义并没有影响：顺序并不是关注点，$w_i$的大小也只会影响音量。

##密度和线性变换
$$p_x\left( x \right) = p_s\left( Wx \right)\left| W \right|$$
其中，$W = A^{-1}$

##ICA算法
$$
W := W+\alpha\left(
\begin{bmatrix}
1-2g\left( w_1^Tx^{\left( i \right)} \right)\\
1-2g\left( w_2^Tx^{\left( i \right)} \right)\\
\vdots \\
1-2g\left( w_n^Tx^{\left( i \right)} \right)
\end{bmatrix}
{x^{\left( i \right)}}^T+\left( W^T \right)^{-1} \right)
$$

#增强学习（Reinforced Learning）
##马可夫决策过程（MDP）
输入为：

* $S$ - **状态**集合
* $A$ - **行动**集合
* $P_{sa}$ - 状态转移分布
    $$\sum_{s'}P_{sa}\left ( s' \right )=1 , P_{sa}\left ( s' \right )\geq0$$
* $\gamma$ - **贴现因子**
    $$0\leq\gamma<1$$
* $R$ - **反馈函数**
    $$R:S\mapsto\mathbb{R}$$

过程建模为：
$$s_0\overset{a_0}{\rightarrow}s_1\overset{a_1}{\rightarrow}s_2\overset{a_2}{\rightarrow}s_3\overset{a_3}{\rightarrow}\dots$$
其中，$s_{t+1}\sim P_{s_t a_t}$

输出为

* $\pi$ - 策略
    $$\pi:S\mapsto A$$
    $$\pi = \arg\max_{\pi}E\left [ R\left ( s_0 \right )+\gamma R\left ( s_1 \right )+\gamma^2 R\left ( s_2 \right )+\dots \right ]$$

##决策过程
对于任意$\pi$，定义价值函数$V^\pi:S\mapsto\mathbb{R}$使得$V^\pi\left( s \right)$是以$s$为初始状态、执行策略$\pi$时反馈总量的数学期望：
$$V^\pi\left ( s \right ) = E\left[ R\left ( s_0 \right )+\gamma R\left ( s_1 \right )+\gamma^2 R\left ( s_2 \right )+\dots \middle| s_0=s,\pi \right]$$
给定$\pi$时，可以看到$V^\pi$是动态规划问题：
$$V^\pi\left( s \right) = R\left( s \right)+\gamma\sum_{s'\in S}P_{s\pi\left( s \right)}\left( s' \right)V^\pi\left( s' \right)$$

定义优化函数$V^*:S\mapsto\mathbb{R}$，并整理：
$$
\begin{aligned}
V^*\left ( s \right ) &= \max_\pi V^\pi\left ( s \right )\\
&=R\left ( s \right )+\max_{a\in A}\gamma\sum_{s'\in S}P_{sa}\left ( s' \right )V^*\left ( s' \right )
\end{aligned}
$$

定义最优策略函数$\pi^*:S\mapsto A$：
$$\pi^*\left ( s \right ) = \arg\max_{a\in A}\sum_{s'\in S}P_{sa}\left ( s' \right )V^*\left ( s' \right )$$

##价值迭代

1.  初始化$\forall s,V\left ( s \right ) = 0$
2.  重复直到收敛：
    对每个$s$，更新：
    $$V\left ( s \right ) := R\left ( s \right )+\max_{a\in A}\gamma\sum_{s'\in S}P_{sa}\left ( s' \right )V\left ( s' \right )$$

重复过程，$V\left( s \right)\to V^*\left( s \right)$。

此处的更新有两种方式：

1. **同步**更新，整体替换，$V := B\left( V \right)$
2. **异步**更新，逐个处理替换，所以同次循环中早前更新的数值用作最新的数值。

最终，$V$会收敛于$V^*$，再使用$\pi^*\left ( s \right ) = \arg\max_{a\in A}\sum_{s'\in S}P_{sa}\left ( s' \right )V^*\left ( s' \right )$可以得到$\pi^*$

##策略迭代

1.  随机初始化$\pi$
2.  重复直到收敛：
    a.  使得$V := V^\pi$
    b.  对于每个$s$，使得
        $$\pi\left ( s \right ) := \arg\max_{a\in A}\sum_{s'\in S}P_{sa}\left ( s' \right )V\left ( s' \right )$$

最终，$V$会收敛于$V^*$，且$\pi$会收敛于$\pi^*$

##$P_{sa}$不确定的情况
每次执行策略$\pi$后通过执行数据重新学习$P_{sa}$

#增强学习
##连续状态
增强学习处理的事务很多情况下是连续变量，而之前的算法是解决离散问题的。

一种方法是将连续的状态切割为多个离散的状态，称**离散化**。离散化的问题在于状态数量过多，一至二维的离散化还能使模型正常运作，而六维以上的模型离散化后几乎无法运作。

##学习模型
将整个模型假想成一个黑盒，前一状态为$s_t$，执行动作$a_t$后，下一状态为一随机变量$s_{t+1} \sim P_{s_{t}a_{t}}$。

要得到这个模型，一种方式是使用物理手动建模。比如杆与小车的问题，可以通过数学方法得到下一时刻的小车状态（欧拉解法即可）。

另外一种方法是通过收集到的数据学习得到一个模型。假定执行$m$次实验，每次实验进行$T$个时间间隔。这个过程可以通过执行随机动作，或者执行某个特定的策略，或者随便其他什么方法。之后可以得到这样的$m$个序列：
$$
\begin{aligned}
s_0^{\left( 1 \right)}\overset{a_0}{\rightarrow}s_1^{\left( 1 \right)}\overset{a_1}{\rightarrow}&s_2^{\left( 1 \right)}\overset{a_2}{\rightarrow}\dots\overset{a_{T-1}}{\rightarrow}s_T^{\left( 1 \right)}\\
s_0^{\left( 2 \right)}\overset{a_0}{\rightarrow}s_1^{\left( 2 \right)}\overset{a_1}{\rightarrow}&s_2^{\left( 2 \right)}\overset{a_2}{\rightarrow}\dots\overset{a_{T-1}}{\rightarrow}s_T^{\left( 2 \right)}\\
&\vdots\\
s_0^{\left( m \right)}\overset{a_0}{\rightarrow}s_1^{\left( m \right)}\overset{a_1}{\rightarrow}&s_2^{\left( m \right)}\overset{a_2}{\rightarrow}\dots\overset{a_{T-1}}{\rightarrow}s_T^{\left( m \right)}
\end{aligned}
$$
然后使用学习算法估计$s_{t+1}$为关于$s_t$和$a_t$的一个函数。

**例子**：假定线性模型，$s_{t+1} = As_t+Ba_t$，取：
$$\arg\min_{A,B}\sum_{i=1}^{m}\sum_{t=0}^{t-1}\left\| s_{t+1}^{\left( i \right)}-\left( As_t^{\left( i \right)}+Ba_t^{\left( i \right)} \right) \right\|^2$$
可以简单地认为这是一个**决定性**模型，或者可以将这个模型视作**随机**模型。
$$s_{t+1} = As_t+Ba_t+\epsilon_t$$
通常，$\epsilon_t \sim \mathcal{N}\left( 0,\Sigma \right)$。

更宽泛地，可以选择$\phi\left( s \right)$来产生更复杂的学习特征。

###适应价值迭代

1. 随机采样$m$个状态$s^{\left( 1 \right)},s^{\left( 1 \right)},\dots,s^{\left( m \right)} \in S$；
2. 初始化$\theta := 0$；
3. 重复：
    a. 对于$i = 1,\dots,m$：
        i. 对于任意行动$a \in A$：
            1) 采样$s_1',\dots,s_k' \sim P_{s^{\left( i \right)}a}$（采用任意MDP模型）；
            2) 设置$q\left( a \right) = \frac{1}{k}\sum_{j=1}^{k}R\left( s^{\left( i \right)} \right)+\gamma V\left( s_j' \right)$；  
                //如此，$q\left( a \right)$为$R\left( s^{\left( i \right)} \right)+\gamma E_{s' \sim P_{s^{\left( i \right)}a}}\left[ V\left( s' \right) \right]$的估计
        ii. 设置$y^{\left( i \right)} = \max_a q\left( a \right)$；  
            //如此，$y^{\left( i \right)}$为$R\left( s^{\left( i \right)} \right)+\gamma\max_a E_{s' \sim P_{s^{\left( i \right)}a}}\left[ V\left( s' \right) \right]$的估计
    b. 设置$\theta := \arg\min_\theta\frac{1}{2}\sum_{i=1}^{m}\left( \theta^T\phi\left( s^{\left( i \right)} \right)-y^{\left( i \right)} \right)^2$

#MDP变种
##状态-行动奖励
奖励函数变为与行动和奖励都相关
$$R : S\times A\mapsto\mathbb{R}$$
整体差别在于必须将奖励函数也放入最大化范畴。

##有限时域MDP
五元组变为$\left( S,A,\left\{ P_{sa} \right\},T,R \right)$。
其中，$T$为时域范围。
下一步分布变为$$s_{t+1} \sim P_{s_ta_t}^{\left( t \right)}$$
价值函数为
$$
\begin{aligned}
V_t^*\left( s \right) &= E\left[ R^{\left( t \right)}\left( s_t,a_t \right)+\dots+R^{\left( T \right)}\left( s_T,a_T \right) \middle| \pi^*,s_t=s \right]\\
&= \max_aR^{\left( t \right)}\left( s_t,a_t \right)+\sum_{s'\in S}P_{sa}^{\left( t \right)}\left( s' \right)V_{t+1}^*\left( s' \right)\\
\pi_t^*\left( s \right) &= \arg\max_aR^{\left( t \right)}\left( s_t,a_t \right)+\sum_{s'\in S}P_{sa}^{\left( t \right)}\left( s' \right)V_{t+1}^*\left( s' \right)
\end{aligned}
$$
整个处理从$V_T^*$和$\pi_T^*$开始，到$V_0^*$和$\pi_0^*$结束。

#线性动力系统
##线性二次调节器（LQR）
给定有限时域MDP问题：$\left( S,A,\left\{ P_{sa} \right\},T,R \right)$，
其中$S \in \mathbb{R}^n$且$A \in \mathbb{R}^d$。

将分布规律建模为
$$P_{s_ta_t}=A_ts_t+B_ta_t+w_t$$
其中：

* $w_t$是零均值的随机变量，并且可以忽略。
* $A_t \in \mathbb{R}^{n\times n}$
* $B_t \in \mathbb{R}^{n\times d}$

设定奖励函数
$$R^{\left( t \right)}\left( s_t,a_t \right)=-\left( s_t^TU_ts_t+a_t^TV_ta_t \right)$$
其中：
$$
\begin{aligned}
U_t &\in \mathbb{R}^{n\times n} &V_t &\in \mathbb{R}^{d\times d}&\\
U_t &\geq 0 &V_t &\geq 0 &\textup{(p.s.d.)}\\
s_t^TU_ts_t &\geq 0 &a_t^TV_ta_t &\geq 0\\
\Rightarrow R^{\left( t \right)}\left( s_t,a_t \right) &\leq 0
\end{aligned}
$$

又通常假定$A$和$B$不随时间变化：
$$
\begin{aligned}
A&=A_1=A_2=A_3=\dots\\
B&=B_1=B_2=B_3=\dots\\
\end{aligned}
$$

鉴于$s_{t+1}=As_t+Ba_t$，执行一系列模拟/实验：
$$
\begin{aligned}
s_0^{\left( 1 \right)}&\overset{a_0^{\left( 1 \right)}}{\rightarrow}s_1^{\left( 1 \right)}\overset{a_1^{\left( 1 \right)}}{\rightarrow}\dots\overset{a_{T-1}^{\left( 1 \right)}}{\rightarrow}s_T^{\left( 1 \right)}\\
&\vdots\\
s_0^{\left( 1 \right)}&\overset{a_0^{\left( 1 \right)}}{\rightarrow}s_1^{\left( 1 \right)}\overset{a_1^{\left( 1 \right)}}{\rightarrow}\dots\overset{a_{T-1}^{\left( 1 \right)}}{\rightarrow}s_T^{\left( 1 \right)}
\end{aligned}
$$
计算$$A,B = \arg\min_{A,B}\frac{1}{2}\sum_{i=1}^{m}\sum_{t=1}^{T-1}\left\| s_{t+1}-\left( As_t+Ba_t \right) \right\|^2$$

如果是非线性系统，则在常规范围内线性化（切线）。
$$
\begin{aligned}
s_{t+1} \approx f\left( \bar{s_t},\bar{a_t} \right)&+\left( \nabla_s f\left( \bar{s_t},\bar{a_t} \right) \right)^T\left( s_t-\bar{s_t} \right)\\
&+\left( \nabla_a f\left( \bar{s_t},\bar{a_t} \right) \right)^T\left( a_t-\bar{a_t} \right)
\end{aligned}
$$
其中，$\bar{s_t}$和$\bar{a_t}$为常规范围（通常系统运行范围）的$s_t$和$a_t$。

于是，$s_{t+1} = As_t+Ba_t$，$s_{t+1}$，$s_t$和$a_t$之间成为线性关系。

$$
\begin{aligned}
V_T^*\left( s_T \right) &= \max_{a_T}R^{\left( T \right)}\left( s_T,a_T \right)\\
 &= \max_{a_T}-s_T^TU_Ts_T-a_T^TV_Ta_T\\
 &= -s_T^TU_Ts_T      &\textup{( 因为 }a_T^TV_Ta_T \geq 0 \textup{)}\\
\pi_T^*\left( s_T \right) &= \arg\max_{a_T}R^{\left( T \right)}\left( s_T,a_T \right)=0
\end{aligned}
$$
而时间回溯得到：
$$V_t^*\left( s_t \right) = \max_{a_t}R^{\left( t \right)}\left( s_t,a_t \right)+E_{s_{t+1}\sim P_{s_ta_t}}\left[ V_{t+1}^*\left( s_{t+1} \right) \right]$$
假定
$$V_{t+1}^*\left( s_{t+1} \right) = s_{t+1}^T\Phi_{t+1}s_{t+1}+\Psi_{t+1}$$
其中$\Phi_{t+1}\in\mathbb{R}^{n\times n},\Psi_{t+1}\in\mathbb{R}$
可以求解$\Phi_t$和$\Psi_t$使得
$$V_t^*\left( s_t \right) = s_t^T\Phi_ts_t+\Psi_t$$
考虑到$V_T^*\left( s_T \right) = -s_t^TU_ts_t$，可知当$\Phi_T=-U_T$，且$\Psi_T=0$时，$V_T^*\left( s_T \right) = s_T^T\Phi_Ts_T+\Psi_T$，符合公式。
而时间回溯得：
$$
\begin{aligned}
V_t^*\left( s_t \right) &= \max_{a_t}R^{\left( t \right)}\left( s_t,a_t \right)+E_{s_{t+1}\sim P_{s_ta_t}}\left[ V_{t+1}^*\left( s_{t+1} \right) \right]\\
&= \max_{a_t}-s_t^T\Phi_ts_t-\Psi_t+E_{s_{t+1}\sim \mathcal{N}\left( A_ts_t+B_ta_t \right)}\left[ s_{t+1}^T\Phi_{t+1}s_{t+1}+\Psi_{t+1} \right]\\
\end{aligned}
$$
应当被简化为关于$a_t$的二次函数。
$$
\begin{aligned}
a_t &= \left( B_t^T\Phi_{t+1}B_t-V_t \right)^{-1}B_t^T\Phi_{t+1}A_t\cdot S_t\\
 &=L_tS_t\\
\pi_t^*\left( s_t \right) &= \arg\max_{a_t}R^{\left( t \right)}\left( s_t,a_t \right)+E_{s_{t+1}\sim P_{s_ta_t}}\left[ V_{t+1}^*\left( s_{t+1} \right) \right]\\
 &= L_tS_t
\end{aligned}
$$

##Riccati方程
$$
\begin{aligned}
V_t^*\left( s_t \right) &= s_t^T\Phi_ts_t+\Psi_t\\
\Phi_t &= A_t^T\left( \Phi_{t+1}-\Phi_{t+1}B_t\left( B_t^T\Phi_{t+1}B_t-V_t \right)^{-1}B_t\Phi_{t+1} \right)A_t-U_t\\
\Psi_t &= -\textup{tr}\Sigma_w\Phi_{t+1}+\Psi_{t+1}
\end{aligned}
$$

##总结

1. 初始化$\Phi_T=-U_T$，$\Psi_T=0$；
2. 使用$\Phi_{t+1}$、$\Psi_{t+1}$求解$\Phi_t$、$\Psi_t$；
3. 使用$\Phi_{t+1}$求解$L_t$；
4. 使用$\pi_t^*\left( s_t \right) = L_tS_t$

#增强学习算法调试
##举例
算法不能很好地控制直升机的情况下可能的情况：

1. 如果算法能够很好地控制模拟器中的直升机，但现实中的飞机却不能很好地控制，
    那么是模拟器的问题；否则：
2. 如果$V^{\pi_\textup{RL}}\left( s_0 \right) < V^{\pi_\textup{human}}\left( s_0 \right)$（算法控制评价劣于人类控制评价），
    那么是算法不够好；否则：
3. 如果$V^{\pi_\textup{RL}}\left( s_0 \right) \geq V^{\pi_\textup{human}}\left( s_0 \right)$（算法控制评价优于人类控制评价），
    那么是评估函数不够好。

仅作例子，实际情况可能不同。

#微分动态规划（DDP）
在每次学习得到一个新的行动模式后重新通过模拟器采集控制轨迹数据并线性化。
尽管$s_{t+1}=f\left( s_t,a_t \right)$并没有改变，但$\bar{s_t}, \bar{a_t}$不同导致了线性化后的内容有所不同。

#卡尔曼滤波（Kalman filter）
实际中，观测结果可能与真实状态有偏差（噪音），卡尔曼滤波正是用于处理这个问题的方法。

设真实状态为$s_t$，观测结果为$y_t$，希望得到$P\left( s_t \middle| y_1,\dots,y_t \right)$。

$s_0,s_1,\dots,s_t,y_1,\dots,y_t$具有一联合高斯分布。
$$
z = \begin{bmatrix}
s_0\\
s_1\\
\vdots\\
s_t\\
y_1\\
\vdots\\
y_t
\end{bmatrix}, z \sim \mathcal{N}\left( \mu , \Sigma \right)
$$
这种方法虽然可以得到结果，但计算量十分巨大。

##过程

1. 预测步骤：
$$s_t|y_1,\dots,y_t \sim \mathcal{N}\left( s_{t|t},\Sigma_{t|t} \right)$$
然后：
$$s_{t+1}|y_1,\dots,y_t \sim \mathcal{N}\left( s_{t+1|t},\Sigma_{t+1|t} \right)$$
其中：
$$
\begin{aligned}
s_{t+1|t} &= As_{t|t}\\
\Sigma_{t+1|t} &= A\Sigma_{t|t}A^T+\Sigma_v
\end{aligned}
$$

2. 更新步骤
$$s_{t+1}|y_1,\dots,y_{t+1} \sim \mathcal{N}\left( s_{t+1|t+1},\Sigma_{t+1|t+1} \right)$$
其中
$$
\begin{aligned}
s_{t+1|t+1} &= s_{t+1|t}+K_{t+1}\left( y_{t+1}-Cs_{t+1|t} \right)\\
K_{t+1} &= \Sigma_{t+1|t}C^T\left( C\Sigma_{t+1|t}C^T+\Sigma_v \right)^{-1}\\
\Sigma_{t+1|t+1} &= \Sigma_{t+1|t}-\Sigma_{t+1|t}C^T\left( C\Sigma_{t+1|t}C^T+\Sigma_v \right)^{-1}C\Sigma_{t+1|t}
\end{aligned}
$$
这里$s_{t+1|t+1}$就是对$s_{t+1}$的最佳估计。

##回到有动作的情况
$$
\begin{aligned}
s_{t+1} &= As_t+Ba_t+w_t &&\bigl( w_t\sim\mathcal{N}\left( 0,\Sigma_w \right) \bigr)\\
y_t &= Cs_t+v_t &&\bigl( v_t\sim\mathcal{N}\left( 0,\Sigma_v \right) \bigr)\\
\end{aligned}
$$

具体步骤：

1. $s_{0|0}=s_0,\Sigma_{0|0}=0\bigl( \text{for }s_0\sim\mathcal{N}\left( s_{0|0},\Sigma_{0|0} \right) \bigr)$
2. 预测：
$$
\begin{aligned}
S_{t+1|t} &= As{t|t}+Ba_t\\
\Sigma_{t+1|t} &= A\Sigma_{t|t}A^T+\Sigma_v\\
a_t &= L_t\cdot s_{t|t}
\end{aligned}
$$

#部分可观察马尔可夫决策过程（POMDP）
问题为七元组：$\left( S,A,Y,\left\{ P_{sa} \right\},\left\{ O_s \right\},T,R \right)$

其中：

* $Y$为可能的观察结果集
* $O_s$为观察分布
    在任何一个步骤，观测结果$y_t \sim O_{s_t}$（如果处于状态$s_t$）

##策略搜索

> 定义一个策略集合$\Pi$，搜索一个好的策略$\pi \in \Pi$。

类比于监督学习：

> 定义一个假设集合$\mathcal{H}$，搜索一个好的假设$h \in \mathcal{H}$。

重新定义：

> 一随机策略为一函数$\pi : S\times A\mapsto\mathbb{R}$  
> $\pi\left( s,a \right)$为在$s$状态下选择动作$a$的可能性。  
> $\sum_a\pi\left( s,a \right) = 1$，$\pi\left( s,a \right) \geq 0$

1. 循环：
    1. 采样$s_0,a_0,s_1,a_1,\dots,s_T,a_T$；
    2. 计算$\textup{payoff} = R\left( s_0,a_0 \right)+\dots+R\left( s_T,a_T \right)$；
    3. 更新$\theta := \theta+\alpha\left[ \frac{\nabla_\theta\pi_\theta\left( s_0,a_0 \right)}{\pi_\theta\left( s_0,a_0 \right)}+\dots+\frac{\nabla_\theta\pi_\theta\left( s_T,a_T \right)}{\pi_\theta\left( s_T,a_T \right)} \right]\times\textup{payoff}$

##

$\hat{s}$为$s$的近似（$\hat{s}$为卡尔曼滤波中的$s_{t|t}$）
$$\pi_\theta\left( \hat{s},a \right) = \frac{1}{1+e^{-\theta^T\hat{s}}}$$

##Pegasus
在模拟中，如果使用随机，会导致模拟结果产生少许不同，影响增强学习算法效果。

反而采用每次不同的随机，则可以使得模拟结果非常接近期望。
