class kNN:
    # 필드 변수 정의
    def __init__(self,k): 
        self.k = k # 이웃의 개수
        self.samples = [] # 훈련 샘플
        self.labels = [] # 훈련 샘플의 라벨
        self.target = [] # 예측할 샘플
    
    # 훈련 샘플과 라벨을 저장
    def fit(self, samples, labels): 
        self.samples = samples # 훈련 샘플
        self.labels = labels # 훈련 샘플의 라벨

    # 예측할 샘플을 저장
    def getTarget(self, target): 
        self.target = target # 예측할 샘플

    # 두 샘플 간의 거리 계산
    def distance(self, a, b):
        d = 0 # 거리 초기화
        for i in range(1, len(a)):
            d += (a[i] - b[i]) ** 2 # 제곱 거리 계산
        return d ** 0.5 # 유클리드 거리 계산

    # 예측 함수
    def predict(self):
        dist = [(self.distance(self.target,self.samples[i]),self.labels[i]) for i in range(len(self.labels))] # 거리 계산 후 리스트 생성
        dist.sort(key=lambda x: x[0]) # 거리 기준으로 정렬
        
        neighbors = [label for (_,label) in dist[:self.k]] # k개의 이웃 라벨 추출

        label_count = {} # 라벨 카운트 초기화
        # 라벨 카운트
        for label in neighbors:
            if label not in neighbors: # 라벨이 없으면 추가
                label_count[label] = 1
            else: # 라벨이 있으면 카운트 증가
                label_count[label] += 1
        
        common = None # 가장 많이 등장한 라벨
        common_count = -1 # 가장 많이 등장한 라벨의 카운트
        # 가장 많이 등장한 라벨 찾기
        for label in label_count:
            # 라벨 카운트 비교
            if label_count[label] > common_count: # 가장 많이 등장한 라벨 업데이트
                common = label
                common_count = label_count[label]
        
        return common # 가장 많이 등장한 라벨 반환
        
#사용 예시
knn = kNN(3) # k=3인 k-NN 객체 생성
sample = [["John", 35, 35, 3],["Rachel", 22, 50, 2],["Hannah", 63, 200, 1], ["Tom", 59, 170, 1], ["Nellie", 25, 40, 4]] # 훈련 샘플 정의
label = [False, True, False, False, True] # 라벨 정의
target = ["David", 37, 50, 2] # 예측할 샘플 정의

knn.fit(sample, label) # 훈련 샘플과 라벨 저장
knn.getTarget(target) # 예측할 샘플 저장
result = knn.predict() # 예측 수행

print(result) # 예측 결과 출력