# 2025 기계학습: scikit-learn 모듈을 사용한 주요 클러스터링 모델 실습
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Iris 분석 클래스 정의
class IrisClusteringAnalyzer:
    def __init__(self):
        self.iris = load_iris()
        self.x = self.iris.data
        self.y = self.iris.target
        self.scaled_x = None
        self.methods = {}
    
    def preprocess(self):
        iris = load_iris()
        X, y = iris.data, iris.target

        # 1) train/test 분할 (예: 7:3)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # 2) 스케일링: train으로 학습 → train/test 모두 변환
        scaler = StandardScaler().fit(self.X_train)
        self.X_train_scaled = scaler.transform(self.X_train)
        self.X_test_scaled  = scaler.transform(self.X_test)

    def cluster_accuracy(y_true, y_pred):
        """
        클러스터 ID별로 가장 많은 실제 라벨을 찾아 매핑(mapping)한 뒤
        accuracy_score를 구함.
        """
        mapped = np.zeros_like(y_pred)
        for cluster in np.unique(y_pred):
            mask = (y_pred == cluster)
            if np.sum(mask)==0: 
                continue
            # 해당 클러스터에서 가장 많은 true label
            majority = mode(y_true[mask]).mode[0]
            mapped[mask] = majority
        return accuracy_score(y_true, mapped)

    def apply_clustering(self):
        kmeans = KMeans(n_clusters=3, random_state=42)
        self.methods["KMeans"] = kmeans.fit_predict(self.scaled_x)

        agg = AgglomerativeClustering(n_clusters=3)
        self.methods["Hierarchical"] = agg.fit_predict(self.scaled_x)

        dbscan = DBSCAN(eps=0.8, min_samples=5)
        dbscan_labels = dbscan.fit_predict(self.scaled_x)
        self.methods["DBSCAN"] = dbscan_labels

        gmm = GaussianMixture(n_components=3, random_state=42)
        gmm_labels = gmm.fit_predict(self.scaled_x)
        self.methods["GMM"] = gmm_labels

        spectral = SpectralClustering(n_clusters=3, affinity='nearest_neighbors', random_state=42)
        self.methods["Spectral"] = spectral.fit_predict(self.scaled_x)

    def plot_clusters(self):
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 3, 1)
        sns.scatterplot(x=self.x[:, 0], y=self.x[:, 1], hue=self.y, palette='tab10')
        plt.title("Ground Truth (Actual Labels)")
        plt.xlabel(self.iris.feature_names[0])
        plt.ylabel(self.iris.feature_names[1])

        for i, (method, labels) in enumerate(self.methods.items()):
            plt.subplot(2, 3, i + 2)
            sns.scatterplot(x=self.x[:, 0], y=self.x[:, 1], hue=labels, palette='tab10')
            plt.title(method)
            plt.xlabel(self.iris.feature_names[0])
            plt.ylabel(self.iris.feature_names[1])
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    analyzer = IrisClusteringAnalyzer()
    analyzer.preprocess()
    analyzer.apply_clustering()
    analyzer.plot_clusters()