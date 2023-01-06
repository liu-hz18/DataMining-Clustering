import numpy as np

class myKMeansModel(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, n_clusters=2, tolerance=0.0001, max_iter=300, random_state=42):
        self.n_clusters = n_clusters
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, data):
        self.centers = {}
        for i in range(self.n_clusters):
            self.centers[i] = data[i]

        for i in range(self.max_iter):
            self.clf = {}
            for i in range(self.n_clusters):
                self.clf[i] = []
            # print("质点:",self.centers)
            for feature in data:
                # distances = [np.linalg.norm(feature-self.centers[center]) for center in self.centers]
                distances = []
                for center in self.centers:
                    # 欧拉距离
                    # np.sqrt(np.sum((features-self.centers[center])**2))
                    distances.append(np.linalg.norm(feature - self.centers[center]))
                classification = distances.index(min(distances))
                self.clf[classification].append(feature)

            # print("分组情况:",self.clf)
            prev_centers = dict(self.centers)
            for c in self.clf:
                self.centers[c] = np.average(self.clf[c], axis=0)

            # '中心点'是否在误差范围
            optimized = True
            for center in self.centers:
                org_centers = prev_centers[center]
                cur_centers = self.centers[center]
                if org_centers.shape != cur_centers.shape:
                    print(org_centers.shape, cur_centers.shape)
                if np.sum((cur_centers - org_centers) / org_centers * 100.0) > self.tolerance:
                # err = (sum([cur_centers[i]  / org_centers[i] for i in range(len(cur_centers))]) - 1) * 100.0
                # if err > self.tolerance:
                    optimized = False
                    break
            if optimized:
                break

    def predict(self, data):
        res = []
        for p_data in data:
            distances = [np.linalg.norm(p_data - self.centers[center]) for center in self.centers]
            res.append(distances.index(min(distances)))
        return res
    
    def fit_predict(self, data):
        self.fit(data)
        return self.predict(data)