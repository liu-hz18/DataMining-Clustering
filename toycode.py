import os, random
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from myKMeans import myKMeansModel
from myDBSCAN import myDBSCANModel

N = 5
DATA_PATH = "./data/cleaned.csv"
SEED = 2333

def set_all_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def Paint(embed, label, title):
    plt.plot()
    plt.scatter(embed[:, 0], embed[:, 1], c=label, s=8, alpha=0.8)
    plt.title(title)
    plt.savefig('{}.png'.format(title))

def mySimilarity(targets, preds):
    N, tot = len(targets), 0
    for i in range(N):
        target, pred = targets[i], preds[i]
        match1 = [_ == target for _ in targets]
        match2 = [_ == pred for _ in preds]
        tot += sum([match1[j] == match2[j] for j in range(N)])
    return tot / N / N

if __name__ == '__main__':
    set_all_seed(SEED)

    original_df = pd.read_csv(DATA_PATH)
    print(len(original_df))
    # 随机抽一小部分, 不然现在 OOM
    original_df = original_df.sample(n=5000, random_state=SEED)
    # 删除前两个无用的ID列
    original_df.drop("encounter_id", axis=1, inplace=True)
    original_df.drop("patient_nbr", axis=1, inplace=True)
    # 训练数据 需要 删除 readmitted 列
    targets = original_df["readmitted"].values
    df = original_df.drop("readmitted", axis=1)
    df.fillna(0, inplace=True)

    # 获得 [n_samples, n_features] 的矩阵
    data = df.values
    print(data.shape)
    # 标准化
    data = preprocessing.scale(data)

    # 可视化
    model = PCA(n_components=2, random_state=SEED)
    # model = TSNE(n_components=2, random_state=SEED, init="pca", learning_rate="auto")
    embed = model.fit_transform(data)
    Paint(embed, targets, 'readmitted')

    preds0 = KMeans(n_clusters=N, random_state=SEED).fit_predict(data)
    # print('Pearson Corr between `readmitted` and `SklearnKMeans`:', np.corrcoef(targets, preds0)[0][1])
    print('Similarity between `readmitted` and `SklearnKMeans`:', mySimilarity(targets, preds0))
    Paint(embed, preds0, 'SklearnKMeans')

    preds1 = myKMeansModel(n_clusters=N, random_state=SEED).fit_predict(data)
    # print('Pearson Corr between `readmitted` and `SklearnKMeans`:', np.corrcoef(targets, preds1)[0][1])
    # print('Pearson Corr between `SklearnKMeans` and `myKMeans`:', np.corrcoef(preds0, preds1)[0][1])
    print('Similarity between `readmitted` and `SklearnKMeans`:', mySimilarity(targets, preds1))
    print('Similarity between `SklearnKMeans` and `myKMeans`:', mySimilarity(preds0, preds1))
    Paint(embed, preds1, 'myKMeans')

    preds2 = DBSCAN(eps=3, min_samples=20).fit_predict(data)
    # print('Pearson Corr between `readmitted` and `SklearnDBSCAN`:', np.corrcoef(targets, preds2)[0][1])
    print('Similarity between `readmitted` and `SklearnDBSCAN`:', mySimilarity(targets, preds2))
    Paint(embed, preds2, 'SklearnDBSCAN')

    preds3 = myDBSCANModel(eps=3, min_samples=20, random_state=SEED).fit_predict(data)
    # print('Pearson Corr between `readmitted` and `myDBSCAN`:', np.corrcoef(targets, preds3)[0][1])
    # print('Pearson Corr between `SklearnDBSCAN` and `myDBSCAN`:', np.corrcoef(preds2, preds3)[0][1])
    print('Similarity between `readmitted` and `myDBSCAN`:', mySimilarity(targets, preds3))
    print('Similarity between `SklearnDBSCAN` and `myDBSCAN`:', mySimilarity(preds2, preds3))
    Paint(embed, preds3, 'myDBSCAN')
    

