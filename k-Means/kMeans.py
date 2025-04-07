import random # 중심점 초기화 할때 필요함 초기 중심점은 랜덤으로 설정
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
    
    # 클러스터 중심점 계산 함수 (클러스터 평균 계산)
    def calculateCentroids(self, clusters):
        n_points = len(clusters) # 포인트의 개수(평균 계산에 사용)
        n_features = len(clusters[0]) # 벡터의 차원 수
        centroids = [] # 클러스터 중심 초기화

        for i in range(n_features):
            feature_sum = 0 # 각 차원에 대한 합 초기화
            for point in clusters:
                feature_sum += point[i] # 각 차원에 대한 합 계산
            centroids.append(feature_sum / n_points) # 평균 계산

        return centroids # 클러스터 중심 반환

    # 클러스터 할당 함수
    def assignClusters(self, data):
        clusters = [[] for _ in range(self.k)] # 클러스터 초기화
        for point in data: # 각 포인트에 대해
            distances = [self.distance(point, centroid) for centroid in self.centroids] # 거리 계산
            min_index = distances.index(min(distances)) # 가장 가까운 클러스터 찾기
            clusters[min_index].append(point) # 클러스터에 포인트 추가

        return clusters # 클러스터 반환
    
    # 클러스터링 수행 함수
    def fit(self, data, init_centroids=None):
        if init_centroids: # 초기 중심점이 주어지면
            self.centroids = init_centroids # 초기 중심점 설정
        else: # 초기 중심점이 주어지지 않으면 랜덤으로 설정
            self.centroids = random.sample(data, self.k)
        
        for _ in range(self.max_iters): # 최대 반복 횟수만큼 반복
            clusters = self.assignClusters(data) # 클러스터 할당
            new_centroids = [] # 새로운 클러스터 중심 초기화

            for cluster in clusters: # 각 클러스터에 대해
                if cluster: # 클러스터가 비어있지 않으면
                    new_centroids.append(self.calculateCentroids(cluster)) # 클러스터 중심 계산
                else: # 클러스터가 비어있으면
                    new_centroids.append(random.choice(data)) # 랜덤으로 포인트 선택
            
            if new_centroids == self.centroids:
                break # 중심점이 변하지 않으면 반복 종료
            self.centroids = new_centroids # 클러스터 중심 업데이트
        return self.centroids # 최종 클러스터 중심 반환
    
    # 예측 함수
    def predict(self, data):
        dist = [] # 거리 초기화
        for centroid in self.centroids:
            dist.append(self.distance(data, centroid))
        return dist.index(min(dist))
    
# 사용 예시
data = [ # 맛집 데이터 [맛, 서비스]
    [1, 3], # 맛집 a
    [2, 2], # 맛집 b
    [2, 3], # 맛집 c
    [4, 4], # 맛집 d
    [4, 5], # 맛집 e
    [5, 4]  # 맛집 f
]

kmeans = KMeans(k=2, max_iters=10) # k=2인 KMeans 객체 생성
kmeans.fit(data) # 클러스터링 수행

for point in data: # 각 포인트에 대해 클러스터 예측 및 출력
    print(f"{point} → Cluster {kmeans.predict(point)}")