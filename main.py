import os
import time
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
import argparse

# for clustering algorithms, see https://scikit-learn.org/stable/modules/clustering.html
# Partitioning Methods: Kmeans
# Graph-Based Methods: AffinityPropagation, SpectralClustering
# Hierarchical Methods: Birch, AgglomerativeClustering
# Density-Based Methods: DBSCAN, OPTICS
# Model-Based Methods: EM(Gaussian Mixture)
from sklearn.cluster import (
    KMeans,
    MiniBatchKMeans,
    BisectingKMeans,
    MeanShift,
    AffinityPropagation,
    SpectralClustering,
    Birch,
    AgglomerativeClustering,
    DBSCAN,
    OPTICS,
)
from sklearn.mixture import (
    GaussianMixture,
    BayesianGaussianMixture
)
# for metrics for clustering, see https://scikit-learn.org/stable/modules/classes.html#clustering-metrics
from sklearn.metrics import (
    # 下列指标需要 label, 似乎用不到
    homogeneity_score,
    completeness_score,
    v_measure_score,
    adjusted_rand_score,
    adjusted_mutual_info_score,
    fowlkes_mallows_score,
    # 下列指标不需要 label
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.metrics.cluster import (
    # 下列指标需要 label, 似乎用不到
    contingency_matrix,
    pair_confusion_matrix,
)

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 全局变量
DATA_PATH = "./data/cleaned.csv"
# 聚类前是否作归一化
NORMALIZE = False
SEED = 2333


def make_model_zoo(n_clusters: int=5):
    return {
        "kmeans": {
            "api": KMeans,
            "normalize": NORMALIZE,
            "args": {
                "n_clusters": n_clusters,
                "random_state": SEED,
            }
        },
        "batch-kmeans": {
            "api": MiniBatchKMeans,
            "normalize": NORMALIZE,
            "args": {
                "n_clusters": n_clusters,
                "random_state": SEED,
            }
        },
        "bisecting-Kmeans": {
            "api": BisectingKMeans,
            "normalize": NORMALIZE,
            "args": {
                "n_clusters": n_clusters,
                "random_state": SEED,
            }
        },
        # "affinity": { # slow
        #     "api": AffinityPropagation,
        #     "normalize": NORMALIZE,
        #     "args": {
        #         "damping": 0.5,
        #         "random_state": SEED,
        #     }
        # },
        # "spectral": { # very slow
        #     "api": SpectralClustering,
        #     "normalize": NORMALIZE,
        #     "args": {
        #         "n_clusters": n_clusters,
        #         "random_state": SEED,
        #     }
        # },
        "Ward": {
            "api": AgglomerativeClustering,
            "normalize": NORMALIZE,
            "args": {
                "n_clusters": n_clusters,
                "linkage": "ward",
            }
        },
        "Agglomerative": {
            "api": AgglomerativeClustering,
            "normalize": NORMALIZE,
            "args": {
                "n_clusters": n_clusters,
                "linkage": "average",
            }
        },
        "Birch": {
            "api": Birch,
            "normalize": NORMALIZE,
            "args": {
                "n_clusters": n_clusters,
            }
        },
        "dbscan": { # `n_clusters` not supported
            "api": DBSCAN,
            "normalize": NORMALIZE,
            "args": {
                "eps": 2,
                "min_samples": 20,
            }
        },
        "optics": { # `n_clusters` not supported
            "api": OPTICS,
            "normalize": NORMALIZE,
            "args": {
            }
        },
        "EM-Gaussian": {
            "api": GaussianMixture,
            "normalize": True,
            "args": {
                "n_components": n_clusters,
                "random_state": SEED,
            }
        },
        "Bayesian-Gaussian": {
            "api": BayesianGaussianMixture,
            "normalize": True,
            "args": {
                "n_components": n_clusters,
                "random_state": SEED,
            }
        }
    }


def set_all_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


# n维霍普金斯统计量计算，input:DataFrame类型的二维数据，output:float类型的霍普金斯统计量
# 默认从数据集中抽样的比例为0.3
def hopkins_statistic(_data: np.ndarray, sampling_ratio: float=0.3) -> float:
    if _data.shape[0] <= 1:
        return 0.5
    # 抽样比例超过0.1到0.5区间任意一端则用端点值代替
    sampling_ratio = np.clip(sampling_ratio, 0.1, 0.5)
    if _data.shape[0] > 10000:
        data = _data[np.random.choice(_data.shape[0], 10000, replace=False), :]
    else:
        data = _data
    num_data = data.shape[0]
    # 抽样数量
    n_samples = int(np.ceil(num_data * sampling_ratio))
    # print("n_samples of hopkins_statistic: ", n_samples)
    # 原始数据中抽取的样本数据
    sample_idxes = np.random.choice(a=num_data, size=n_samples, replace=False)
    sample_data = data[sample_idxes, :]
    # 原始数据抽样后剩余的数据
    other_idxes = np.setdiff1d(np.arange(num_data), sample_idxes, assume_unique=True)
    other_data = data[other_idxes, :]
    # data = data.drop(index=sample_data.index) #,inplace = True)
    # 原始数据中抽取的样本与最近邻的距离之和
    data_dist = cdist(other_data, sample_data).min(axis=0).sum()
    # 人工生成的样本点，从平均分布中抽样(artificial generate samples)
    ags_data = []
    for col in range(data.shape[1]):
        ags_data.append(np.random.uniform(data[:, col].min(), data[:, col].max(), n_samples))
    ags_data = np.array(ags_data).T
    # 人工样本与最近邻的距离之和
    ags_dist = cdist(data, ags_data).min(axis=0).sum()
    # 计算霍普金斯统计量H
    H_value = ags_dist / (data_dist + ags_dist)
    return H_value


def hopkins_score(data: np.ndarray, labels: np.ndarray, sampling_ratio: float=0.3, reduction: str="weighted"):
    # 抽样比例超过0.1到0.5区间任意一端则用端点值代替
    sampling_ratio = min(max(sampling_ratio, 0.1), 0.5)
    # 对于每一类，分别计算 hopkins statistic
    groups = np.unique(labels)
    scores = []
    nums = []
    for g in groups:
        group_data = data[np.where(labels == g)]
        hopkins = hopkins_statistic(group_data, sampling_ratio=sampling_ratio)
        scores.append(hopkins)
        nums.append(group_data.shape[0])
    H = 0.0
    if reduction == "weighted":
        weights = np.array(nums) / data.shape[0]
        H = np.average(scores, weights=weights)
    elif reduction == "mean":
        H = np.mean(scores)
    else:
        raise NotImplementedError
    return H


def get_metrics(data, labels):
    if data.shape[0] > 10000:
        _sampled_index = np.random.choice(data.shape[0], 10000, replace=False)
        data = data[_sampled_index, :]
        labels = labels[_sampled_index]
    return {
        "silhouette": silhouette_score(data, labels),
        "calinski_harabas": calinski_harabasz_score(data, labels),
        "davies_bouldin": davies_bouldin_score(data, labels),
        "hopkins": hopkins_score(data, labels, sampling_ratio=0.3, reduction='weighted')
    }


# 数据降维，方便可视化
def embed(data, embed_dim: int=2, method: str="tsne", normalize: bool=False):
    if normalize:
        data = StandardScaler().fit_transform(data)
    if method == "tsne":
        model = TSNE(n_components=embed_dim, random_state=SEED, init="pca", learning_rate="auto")
        embed = model.fit_transform(data)
    elif method == "pca":
        model = PCA(n_components=embed_dim, random_state=SEED)
        embed = model.fit_transform(data)
    else:
        raise NotImplementedError(f"embedding method {method} NOT supported.")
    return embed


def visualize_cluster(embed_data, label, title: str, path):
    plt.clf()
    plt.plot()
    plt.scatter(embed_data[:, 0], embed_data[:, 1], c=label, s=8, alpha=[0.8 if l != -1 else 0.1 for l in label])
    plt.title(title)
    plt.savefig(f"results/{path}")


# 每个类的数量的直方图
def cluster_size_histogram(results):
    df = {
        "model": [],
        "cluster": [],
        "count": [],
    }
    for model_name, values in results.items():
        labels = values["label"]
        groups = np.unique(labels)
        w = np.sort(np.array([np.sum(labels == g) for g in groups]))[::-1]
        df["model"].extend([model_name] * len(groups))
        df["cluster"].extend(groups)
        df["count"].extend(w.tolist())
    df = pd.DataFrame(df)
    print(df.head())
    plt.clf()
    sns.barplot(
        data=df,
        x="model",
        y="count",
        hue="cluster",
    )
    plt.xticks(rotation=20)
    plt.title(f"cluster sizes of different models")
    plt.savefig(f"results/cluster_sizes.png")


# 和 readmitted 列对比得到heatmap
def readmitted_heatmap(original_df, results):
    for model_name, values in results.items():
        labels = values["label"]
        cluster_group = np.unique(labels)
        readmitted_group = np.unique(original_df["readmitted"].values)
        corr_mat = []
        for cluster in cluster_group:
            corr_list = []
            for readmitted in readmitted_group:
                cluster_idxes = np.where(labels == cluster)
                readmitted_idxes = np.where(original_df["readmitted"].values == readmitted)
                corr_list.append(len(np.intersect1d(cluster_idxes, readmitted_idxes)))
            corr_list = np.array(corr_list) / sum(corr_list)
            corr_mat.append(corr_list)
        corr_mat = np.array(corr_mat).T
        plt.clf()
        sns.heatmap(data=corr_mat, vmin=0.0, vmax=1.0, cmap=sns.cm.rocket_r, annot=True, fmt=".2f", yticklabels=["<30", ">30", "NO"])
        plt.xlabel("Cluster")
        plt.ylabel("Readmitted")
        plt.title(f"readmitted records in each cluster ({model_name})")
        plt.savefig(f"results/readmitted/{model_name}.png")


def readmitted_corr(original_df, sampled_index, results):
    readmitted = original_df["readmitted"].values
    readmitted = readmitted[sampled_index]
    corr_list = []
    for _, values in results.items():
        labels = values["label"]
        corr = np.corrcoef(readmitted, labels)[0][1]
        corr_list.append(corr)
    plt.clf()
    plt.bar(x=results.keys(), height=corr_list)
    plt.title("Pearson Corr between `readmitted` and `cluster` on different models")
    plt.savefig("results/readmitted/corr.png")


# 不同模型在不同cluster数量下的分数
def ablation_over_cluster_num(data):
    df = {
        "model": [],
        "clusters": [],
        "silhouette": [],
        "calinski_harabas": [],
        "davies_bouldin": [],
        "hopkins(weighted)": [],
    }
    for n_cluster in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        model_zoo = make_model_zoo(n_cluster)
        for m in model_zoo.keys():
            if m == 'dbscan' or m == 'optics':
                continue
            print(f"Cluster {n_cluster} on model {m}")
            if model_zoo[m]["normalize"]:
                scaled_data = StandardScaler().fit_transform(data)
            else:
                scaled_data = data
            labels = model_zoo[m]["api"](**model_zoo[m]["args"]).fit_predict(scaled_data)
            metrics = get_metrics(scaled_data, labels)
            df["model"].append(m)
            df["clusters"].append(n_cluster)
            df["silhouette"].append(metrics["silhouette"])
            df["calinski_harabas"].append(metrics["calinski_harabas"])
            df["davies_bouldin"].append(metrics["davies_bouldin"])
            df["hopkins(weighted)"].append(metrics["hopkins"])
    df = pd.DataFrame(df)
    for metric in ["silhouette", "calinski_harabas", "davies_bouldin", "hopkins(weighted)"]:
        plt.clf()
        sns.lineplot(data=df, x="clusters", y=metric, hue="model")
        plt.title(f"`{metric}` score over different models and cluster nums")
        plt.legend(loc='upper right')
        plt.savefig(f"results/cluster_num/{metric}.png")


def apply(data, sampled_index, n_clusters: int=5):
    df = {
        "model": [],
        "time": [],
        "silhouette": [],
        "calinski_harabas": [],
        "davies_bouldin": [],
        "hopkins(weighted)": [],
    }
    results = {}
    model_zoo = make_model_zoo(n_clusters)
    for m in model_zoo.keys():
        print(m)
        if model_zoo[m]["normalize"]:
            scaled_data = StandardScaler().fit_transform(data)
        else:
            scaled_data = data
        begin_timestamp = time.time()
        labels = model_zoo[m]["api"](**model_zoo[m]["args"]).fit_predict(scaled_data)
        end_timestamp = time.time()
        results[m] = {
            "label": labels,
            "normalize": model_zoo[m]["normalize"],
        }
        print(f"time elapsed={end_timestamp - begin_timestamp}")
        print(f"silhouette_score={silhouette_score(scaled_data, labels)}")
        print(f"calinski_harabaz_score={calinski_harabasz_score(scaled_data, labels)}")
        print(f"davies_bouldin_score={davies_bouldin_score(scaled_data, labels)}")
        print(f"hopkins_score={hopkins_score(scaled_data, labels, sampling_ratio=0.3, reduction='weighted')}")
        print("")
    df = pd.DataFrame(df)
    df.to_csv("metrics.csv", index=False)
    return results


if __name__ == '__main__':
    set_all_seed(SEED)
    plt.style.use('ggplot')

    original_df = pd.read_csv(DATA_PATH)
    # 随机抽一小部分, 不然现在 OOM
    original_df = original_df.sample(n=10000, random_state=SEED)
    # 删除前两个无用的ID列
    original_df.drop("encounter_id", axis=1, inplace=True)
    original_df.drop("patient_nbr", axis=1, inplace=True)
    # 删除几个数据缺失率太高的列
    original_df.drop("weight", axis=1, inplace=True)
    original_df.drop("payer_code", axis=1, inplace=True)
    original_df.drop("medical_specialty", axis=1, inplace=True)
    # 训练数据 需要 删除 readmitted 列
    df = original_df.drop("readmitted", axis=1)
    df.fillna(0, inplace=True)

    # 获得 [n_samples, n_features] 的矩阵
    raw_data = df.values

    if NORMALIZE:
        data = StandardScaler().fit_transform(raw_data)
    else:
        data = raw_data
    # 聚类前的 hopkins_statistic
    print(f"hopkins_statistic={hopkins_statistic(data, sampling_ratio=0.3)}")

    # 跑一遍各种算法
    results = apply(data, n_clusters=5)

    # 不同算法 每类容量 的统计图
    cluster_size_histogram(results)

    # 回答 readmitted 和 聚类 关联性 的问题
    readmitted_corr(original_df, results)
    readmitted_heatmap(original_df, results)

    # 降维+聚类结果可视化
    embed_pca = embed(raw_data, method="pca", normalize=False)
    embed_pca_norm = embed(raw_data, method="pca", normalize=True)
    print(raw_data.shape)
    print(embed_pca.shape)
    for method, values in results.items():
        visualize_cluster(embed_pca_norm if values["normalize"] else embed_pca, values["label"], title=f"{method}(PCA)")

    embed_tsne = embed(raw_data, method="tsne", normalize=False)
    embed_tsne_norm = embed(raw_data, method="tsne", normalize=True)
    print(raw_data.shape)
    print(embed_tsne.shape)
    for method, values in results.items():
        visualize_cluster(embed_tsne_norm if values["normalize"] else embed_tsne, values["label"], title=f"{method}(t-SNE)")

    # 不同算法、聚类个数的 ablation，为加速，只抽样10000个样本
    ablation_over_cluster_num(data)
