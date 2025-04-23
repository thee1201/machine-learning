# 2025 기계학습: scikit-learn 모듈을 사용한 주요 클러스터링 모델 실습
from sklearn.datasets import load_iris  # Iris 데이터셋 불러오기 함수
from sklearn.model_selection import train_test_split  # 학습/테스트 데이터 분할 함수
from sklearn.preprocessing import StandardScaler  # 데이터 표준화(스케일링) 클래스
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering  # 주요 클러스터링 알고리즘
from sklearn.mixture import GaussianMixture  # 가우시안 혼합 모형(EM 알고리즘)
from sklearn.metrics import accuracy_score  # 정확도 계산 함수
from sklearn.metrics import silhouette_score  # 실루엣 점수 계산 함수 (사용 예정)
from scipy.stats import mode  # 최빈값 계산 함수
import matplotlib.pyplot as plt  # 시각화 라이브러리
import seaborn as sns  # 통계적 시각화 라이브러리
import pandas as pd  # 데이터 처리 라이브러리
import numpy as np  # 수치 연산 라이브러리

# Iris 분석 클래스 정의
class IrisClusteringAnalyzer:
    def __init__(self):
        self.iris = load_iris()  # Iris 데이터셋 로드
        self.x = self.iris.data  # 특성(4개 속성) 배열
        self.y = self.iris.target  # 레이블(종 클래스) 배열
        self.scaled_x = None  # 스케일된 데이터 초기화
        self.methods = {}  # 클러스터링 결과 저장 dict
        self.accuracies = {}  # 정확도 저장 dict
    
    def preprocess(self):
        iris = load_iris()  # 다시 Iris 데이터셋 로드
        X, y = iris.data, iris.target  # 특성, 레이블 분리

        # 1) train/test 분할 (예: 7:3)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y  # stratify로 클래스 비율 유지
        )

        # 2) 스케일링: train으로 학습 → train/test 모두 변환
        scaler = StandardScaler().fit(self.X_train)  # 학습 데이터로 스케일러 학습
        self.X_train_scaled = scaler.transform(self.X_train)  # 학습 데이터 변환
        self.X_test_scaled  = scaler.transform(self.X_test)  # 테스트 데이터 동일 변환

    @staticmethod
    def cluster_accuracy(y_true, y_pred):
        """
        클러스터 ID별로 가장 많은 실제 라벨을 찾아 매핑(mapping)한 뒤
        accuracy_score를 구함.
        """
        mapped = np.zeros_like(y_pred)  # 예측 라벨 매핑 배열 초기화
        for cluster in np.unique(y_pred):  # 각 클러스터 ID 순회
            mask = (y_pred == cluster)  # 해당 클러스터 인덱스 추출
            if not mask.any():  # 빈 클러스터 건너뛰기
                continue
            # 해당 클러스터에서 가장 많은 true label을 계산
            labels, counts = np.unique(y_true[mask], return_counts=True)
            majority = labels[counts.argmax()]
            mapped[mask] = majority  # 매핑된 다수결 레이블 지정
        return accuracy_score(y_true, mapped)  # 정확도 반환

    def apply_clustering(self):
         # --- KMeans: train→test 예측 & Accuracy 계산
        kmeans = KMeans(n_clusters=3, random_state=42)  # KMeans 객체 생성
        kmeans.fit(self.X_train_scaled)  # 학습 데이터로 클러스터링 학습
        km_labels = kmeans.predict(self.X_test_scaled)  # 테스트 데이터 클러스터 예측
        self.methods["KMeans"]    = km_labels  # 결과 저장
        self.accuracies["KMeans"] = self.cluster_accuracy(self.y_test, km_labels)  # 정확도 저장

        # --- Hierarchical on test set
        agg = AgglomerativeClustering(n_clusters=3)  # 계층적 클러스터링 객체
        ag_labels = agg.fit_predict(self.X_test_scaled)  # 테스트 데이터 직접 예측
        self.methods["Hierarchical"]    = ag_labels  # 결과 저장
        self.accuracies["Hierarchical"] = self.cluster_accuracy(self.y_test, ag_labels)  # 정확도 저장

        # --- DBSCAN on test set
        dbscan = DBSCAN(eps=0.8, min_samples=5)  # 밀도 기반 클러스터링 객체
        db_labels = dbscan.fit_predict(self.X_test_scaled)  # 테스트 데이터 직접 예측
        self.methods["DBSCAN"]    = db_labels  # 결과 저장
        self.accuracies["DBSCAN"] = self.cluster_accuracy(self.y_test, db_labels)  # 정확도 저장

        # --- GMM (EM): train→test 예측
        gmm = GaussianMixture(n_components=3, random_state=42)  # GMM 객체 생성
        gm_labels = gmm.fit(self.X_train_scaled).predict(self.X_test_scaled)  # 예측
        self.methods["GMM"]    = gm_labels
        self.accuracies["GMM"] = self.cluster_accuracy(self.y_test, gm_labels)

        # --- Spectral on test set
        spectral = SpectralClustering(
            n_clusters=3, affinity='nearest_neighbors', random_state=42
        )  # 스펙트럴 클러스터링 객체
        sp_labels = spectral.fit_predict(self.X_test_scaled)  # 예측
        self.methods["Spectral"]    = sp_labels
        self.accuracies["Spectral"] = self.cluster_accuracy(self.y_test, sp_labels)

        # 모든 모델 돌리고 나서 정확도 출력
        print("=== Clustering Accuracies ===")
        for method, acc in self.accuracies.items():  # 딕셔너리 순회
            print(f"{method:15s}: {acc:.2f}")

    def plot_clusters(self):
        plt.figure(figsize=(15, 10))  # 전체 그림 크기 설정

        # 1) 테스트 세트 실제 라벨
        plt.subplot(2, 3, 1)  # 2x3 그리드 중 첫 번째 서브플롯
        sns.scatterplot(
            x=self.X_test_scaled[:, 0],  # 첫 번째 특성(x축)
            y=self.X_test_scaled[:, 1],  # 두 번째 특성(y축)
            hue=self.y_test,  # 실제 레이블로 색상 구분
            palette='tab10',  # 10가지 색상 팔레트
            legend=False  # 범례 간단히 처리
        )
        plt.title("Ground Truth (Test Labels)")  # 제목
        plt.xlabel(self.iris.feature_names[0])  # x축 레이블
        plt.ylabel(self.iris.feature_names[1])  # y축 레이블

        # 2) 각 클러스터링 결과도 동일하게 테스트 세트에 대해
        for i, (method, labels) in enumerate(self.methods.items()):  # 결과 딕셔너리 순회
            plt.subplot(2, 3, i + 2)  # 서브플롯 인덱스
            sns.scatterplot(
                x=self.X_test_scaled[:, 0],
                y=self.X_test_scaled[:, 1],
                hue=labels,  # 예측 클러스터로 색 구분
                palette='tab10',
                legend=False
            )
            plt.title(method)  # 클러스터링 방법 이름
            plt.xlabel(self.iris.feature_names[0])
            plt.ylabel(self.iris.feature_names[1])

        plt.tight_layout()  # 레이아웃 간격 조정
        plt.show()  # 그래프 출력
    
    def tune_dbscan(self, eps_values, min_samples_values):
        """
        eps / min_samples 조합별로 DBSCAN test-set 정확도 계산,
        결과를 DataFrame으로 반환
        """
        records = []  # 결과 리스트 초기화
        for eps in eps_values:  # eps 값 순회
            for ms in min_samples_values:  # min_samples 값 순회
                labels = DBSCAN(eps=eps, min_samples=ms).fit_predict(self.X_test_scaled)  # DBSCAN 객체 생성 및 테스트 데이터 예측
                acc = self.cluster_accuracy(self.y_test, labels)  # 정확도 계산
                records.append({"eps": eps, "min_samples": ms, "accuracy": acc})  # 기록 추가
        return pd.DataFrame(records)  # DataFrame으로 반환

if __name__ == "__main__":
    analyzer = IrisClusteringAnalyzer()  # 분석기 인스턴스 생성
    analyzer.preprocess()  # 전처리 수행
    analyzer.apply_clustering()  # 클러스터링 실행 & 정확도 출력
    analyzer.plot_clusters()  # 결과 시각화