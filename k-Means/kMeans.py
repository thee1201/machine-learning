class KMeans:
    # 필요한 매개변수: k (클러스터 수), max_iters (최대 반복 횟수) 
    def __init__(self, k, max_iters):
        self.k = k # 클러스터 수
        self.max_iters = max_iters # 최대 반복 횟수
        self.centroids = [] # 클러스터 중심 초기화
    
    # 거리 계산 함수
    def distance(self, a, b):
        d = 0 # 거리 초기화
        for i in range(len(a)):
            d += (a[i] - b[i]) ** 2 # 제곱 거리 계산
        return d ** 0.5 # 유클리드 거리 계산
    
    # 클러스터 중심점 계산 함수 (평균 계산)
    def calculate_centroids(self, clusters):
        


