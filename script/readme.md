# VE445 project

## Method

In this project, I use gcn to handle the molecule

+ Since the molecule is heterogeneous graph, I used onhot style to represent the feature, which includes what's tha atom and is it aromatic. For edge infomation, since simply implemented the edge information into the weight in the adjacent matrix will have negative impact on the F1-score, I just ignore edge information here, which can be improved by more fancy method.

+ The Laplacian matrix I use here is the random walk one, which is $D^{-1}A$, where $D$ is the degree matrix and $A$ is the adjacent matrix. 

## Usage

+ If you didn't process data before, when you first run "./test.py" or "./train.py", it will process data automatically. Thus, it may take some time.

+ The train.py will replace previous trained model, if you want keep previous one, please change the name of saved model in "./train.py"

+ The process of data needs library "pysmiles" and "rdkit", to install

```{python}
pip install pysmiles
conda install -c conda-forge rdkit
```

