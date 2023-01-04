# Data Mining HW2: Clustering

### preparation
```
pip install -r requirements.txt
```

### preprocess
Convert string features to discrete numeric features.
```
python preprocess.py
```

### Enable Intel CPU optimizations
```
from sklearnex import patch_sklearn, unpatch_sklearn
patch_sklearn()

from sklearn.cluster import ...
```

### Train and Plot
```
python main.py
```

### Results

