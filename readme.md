# VE445 project

### Name on Kaggle

My name is Junjzhang, actually, first I use Bufan He, after I see the anouncement regarding kaggle name, I create a new one and submits following work on that account. I feel so sorry about the inconvience caused by me.

### Description

In this project, GCN is used to handle the molecule.


#### Graph embedding

+ Since the molecule is heterogeneous graph, I used onhot style to represent the feature with size $(83,1)$, which includes what's tha atom and is it aromatic. For edge infomation, I just convert the bond information into the weight in the adjacent matrix.

+ The Laplacian matrix I use here is the random walk one, which is $\hat{D}^{-1}\hat{A}$, where $\hat{D}$ is the degree matrix with self-loop and $\hat{A}$ is the adjacent matrix with self loop.

#### Network structrue

+ Convolution layer 1 :(83,60) relu 

+ Convolution layer 2: (60,45) relu

+ Convolution layer 3: (45,35) relu

+ Convolution layer 4: (35,30) relu

  + reduce by sum by each features

+ Full connected layer 1: (30,30) relu

+ Full connected layer 2: (30,30) tanh

+ output layer: (30,2) softmax

#### Union metric

+ Union $accuracy = 0.9116218457528729$

+ Union $banlanced accuracy = 0.6543738895188378$

+ Union $F1-score = 0.650564170843316$

#### Traning technique

+ Since the dataset is inbanlanced, The feeded batch is consists of 17 positive data points and 33 negative data points, the number of positive data points will grow as epoch grows.

+ Learning rate decay is used.

+ Early drop since I found that good performance on validation set will have a relative bad performance on test set. This main casued by overfitting.

+ Dropout is used at first and be abandoned soon because I found that since the number of hidden layer is not that much, the dropout will make it hard to converge.

+ L2 penelty is used on full connected layer to prevent overfitting.

### Usage

+ Since GCN is used here, the original data will be converted into graph first.

+ If you didn't process data before, when you first run "./test.py" or "./train.py", it will process data automatically. Thus, it may take some time.

+ The train.py will replace previous trained model, if you want keep previous one, please change the name of saved model in "./train.py".

+ The process of data needs library "pysmiles" and "rdkit", to install:

```{python}
pip install pysmiles
conda install -c conda-forge rdkit
```

