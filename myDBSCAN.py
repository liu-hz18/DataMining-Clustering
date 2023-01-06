import random
class myDBSCANModel:
    def __init__(self, eps=0.1, min_samples=10, random_state=42):
        self.eps = eps
        self.min_samples = min_samples
        random.seed(random_state)
    
    #获取一个点的ε-邻域（记录的是索引）
    def getNeibor(self, point, dataset, eps):
        res = []
        for i in range(len(dataset)):
            if sum((point - dataset[i]) ** 2) < eps:
                res.append(i)
        return res

    def fit_predict(self, data):
        coreObjs = {}
        C = {}
        n = len(data)
        #找出所有核心对象，key是核心对象的index，value是ε-邻域中对象的index
        for i in range(n):
            neibor = self.getNeibor(data[i], data, self.eps ** 2)
            if len(neibor) >= self.min_samples:
                coreObjs[i] = neibor
        oldCoreObjs = coreObjs.copy()
        k = 0 #初始化聚类簇数
        
        notAccess = list(range(n)) #初始化未访问样本集合（索引）
        while len(coreObjs) > 0:
            OldNotAccess = []
            OldNotAccess.extend(notAccess)
            cores = list(coreObjs.keys())
            #随机选取一个核心对象
            randNum = random.randint(0, len(cores))
            core = cores[randNum]
            queue = []
            queue.append(core)
            notAccess.remove(core)
            while len(queue)>0:
                q = queue[0]
                del queue[0]
                if q in oldCoreObjs:
                    delte = [val for val in oldCoreObjs[q] if val in notAccess] # Δ = N(q) ∩ Γ
                    queue.extend(delte)#将Δ中的样本加入队列Q
                    notAccess = [val for val in notAccess if val not in delte] # Γ = Γ \ Δ
            k += 1
            C[k] = [val for val in OldNotAccess if val not in notAccess]
            for x in C[k]:
                if x in coreObjs.keys():
                    del coreObjs[x]

        res = [-1 for i in range(n)]
        for key, val in C:
            for _ in val:
                res[_] = key
        return res

