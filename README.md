# iNeuron Machine Learming
1. [**Linear Models :**](https://github.com/MohammadWasiq0786/iNeuron-Machine-Learming-/tree/main/Linear%20Models)

Here I have discussed as follows :
* Theory of Linear Models
* Simple Linear Regression
* Multiple Linear Regression
* Multi-Collinearity
* Ridge Regression
* Lasso Regression

Used Libraries : 
```python
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LinearRegression
import pickle
import statsmodels.api as sm 
import statsmodels.formula.api as smf
from sklearn.linear_model  import Ridge, Lasso, RidgeCV, LassoCV, ElasticNet, ElasticNetCV, LinearRegression
sns.set()
```

2. [**Logistic Regression :**](https://github.com/MohammadWasiq0786/iNeuron-Machine-Learming/tree/main/Logistic%20Regression)

Here I have discussed as follows :
* Theory of Logistic Regression
* Logistic Regression Implementation
* VIF (Variance Inflation Factor)
* Multi-Collinearity
* Ridge Regression
* Lasso Regression
* Accuracy 
* Confusion Matrix 
* ROC Curve 
* ROC AUC Score

Used Libraries : 
```PYTHON
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model  import Ridge, Lasso, RidgeCV, LassoCV, ElasticNet, ElasticNetCV, LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor 
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import scikitplot as skl
sns.set()
```
