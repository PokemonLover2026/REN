import copy
import numpy as np
from sklearn.cluster import KMeans


class Delta_op:

    def __init__(self, L, L_majo, L_mino, specifications, loc, pval, idx):
        self.L_majo =  L_majo
        self.L_mino = L_mino
        self.L_q = L.L_q
        # L.pure_lyapunov should be a len = self.pure_samples_count list
        self.pure_lyapunov = copy.deepcopy(L.pure_lyapunov)
        self.rho = specifications.rho
        self.sample_count = L.pure_sample_count
        self.data = None
        self.loc = loc
        self.pval = pval
        self.idx_pure = idx
        self.delta = self.cal_delta_op()

    def L_KM(self, L):
        # 创建KMeans对象
        n_clusters = 2
        kmeans = KMeans(n_clusters=n_clusters)
        # cluster_point_list = []

        # 对数据进行拟合

        data = L.reshape(-1,1)
        if data.shape[0] > 1:
            kmeans.fit(data)

            # 获取聚类中心
            cluster_center = kmeans.cluster_centers_

            # 预测每个数据点的簇标签
            labels = kmeans.predict(data)
            label_counts = np.bincount(labels)
            most_frequent_label = np.argmax(label_counts)
            # 打印聚类中心
            print("Cluster center:\n", cluster_center)

            self.data = data[labels == most_frequent_label]
            # cluster_points = data[labels == most_frequent_label]
            # cluster_point = np.amax(self.data)
            cluster_point = cluster_center[most_frequent_label]
            # cluster_point_list.append(cluster_point)
        else:
            cluster_point = data[0]

        return cluster_point

    def amax_Lq(self, L):

        L = np.array(L)
        peak = self.L_KM(L)
        max_L = np.array(peak)

        return max_L

    def cal_delta_op(self):
        # The len of delta_op_list is related with self.sample_count

        if self.L_majo is not None:
            if self.pval > 0.1:
                # delta_op_list = []

                pure_lyapunov = self.pure_lyapunov[self.idx_pure]
                delta_op = (self.rho - pure_lyapunov) / self.loc
                delta_op = delta_op.tolist()
                delta_op = delta_op[0]
                print(f"The used max loc is {self.loc}.")
                # delta_op_list.append(delta_op)
            else:
                # delta_op_list = []
                max_Lq = self.amax_Lq(self.L_majo) # distinguish different samples with function amax_Lq()

                pure_lyapunov = self.pure_lyapunov[self.idx_pure]
                delta_op = (self.rho - pure_lyapunov) / max_Lq
                delta_op = delta_op.tolist()
                delta_op = delta_op[0]
                print(f"The used max L_q is {max_Lq}.")
                # delta_op_list.append(delta_op)
        else:
            # mino-group:
            if self.pval > 0.01 and self.L_mino.shape[0] > len(self.L_q[self.idx_pure])//3:
                # delta_op_list = []

                pure_lyapunov = self.pure_lyapunov[self.idx_pure]
                try:
                    delta_op = (self.rho - pure_lyapunov) / self.loc
                except:
                    delta_op = (self.rho - pure_lyapunov.float()) / self.loc
                delta_op = delta_op.tolist()
                delta_op = delta_op[0]
                print(f"The used max loc is {self.loc}.")
                # delta_op_list.appeuund(delta_op)
            else:
                # delta_op_list = []
                max_Lq = self.amax_Lq(self.L_mino)

                pure_lyapunov = self.pure_lyapunov[self.idx_pure]
                delta_op = (self.rho - pure_lyapunov) / max_Lq
                delta_op = delta_op.tolist()
                delta_op = delta_op[0]
                print(f"The used max L_q is {max_Lq}.")
                # delta_op_list.append(delta_op)


        return delta_op
