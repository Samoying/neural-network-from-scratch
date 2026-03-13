# 계산 그래프를 이용한 신경망 구현

이 저장소에는 NumPy를 사용하여 Python으로 구현된 기본적인 신경망이 있습니다.

파일은 데이터 셋에 맞춰 3가지로 구분해 놓았습니다.

데이터 셋은 총 세 개가 있으며, 각각 `XOR`, `circle`, `spiral`입니다.
### XOR

<img width="547" height="413" alt="Image" src="https://github.com/user-attachments/assets/394bf525-faea-43dc-85d8-1aa08e498667" />

학습 후 결정 경계를 시각화한 모습

<img width="583" height="418" alt="Image" src="https://github.com/user-attachments/assets/b50d7063-7a4b-4b75-b22f-b1d7c500a8f1" />

<img width="583" height="418" alt="Image" src="https://github.com/user-attachments/assets/cf3b2ce7-5dcb-4ac5-bf69-091c7ecea7fd" />

### circle

<img width="559" height="413" alt="Image" src="https://github.com/user-attachments/assets/3719ea54-27e9-4edd-9104-8a14b5b497a2" />

학습 후 결정 경계를 시각화한 모습

<img width="515" height="389" alt="Image" src="https://github.com/user-attachments/assets/0908e508-05d0-41f4-8657-f76dfcb2cd6f" />

## spiral

<img width="567" height="413" alt="Image" src="https://github.com/user-attachments/assets/a34f8fcd-d103-4765-bf59-72aa8e4e97b7" />

학습 후 결정 경계를 시각화한 모습

<img width="515" height="389" alt="Image" src="https://github.com/user-attachments/assets/72ef0219-d0c9-4bb0-ac4f-2b93bfb2bfa5" />

<img width="515" height="389" alt="Image" src="https://github.com/user-attachments/assets/a6e63538-ac70-44d2-9d22-2f760298a52e" />

<img width="640" height="480" alt="Image" src="https://github.com/user-attachments/assets/51cd6f1d-d048-42d8-ba4d-4b3be355760e" />

## 프로젝트 설명

이 프로젝트는 이진 분류가 가능한 간단한 피드포워드 신경망을 보여줍니다. 시그모이드(Sigmoid) 및 ReLU 활성화 함수와 같은 일반적인 신경망 구성 요소를 직접 구현했으며, 네트워크 아키텍처, 순전파(forward pass) 및 역전파(backpropagation)를 캡슐화하는 `Net` 클래스가 포함되어 있습니다.

## 구현된 구성 요소

### 네트워크 클래스 (`Layer`)

이 프로젝트는 순수 Python과 NumPy 라이브러리를 사용하여 구현된 간단한 신경망을 보여줍니다. 이 신경망은 다음을 포함한 사용자 정의 레이어로 구성됩니다:

#### 계산 그래프
*   `AddLayer`: 덧셈 연산을 수행합니다.
*   `MulLayer`: 곱셈 연산을 수행합니다.

#### 활성화 함수

*   `SigLayer`: 출력 계층을 위한 시그모이드 활성화 함수와 그 미분을 구현합니다.
*   `ReluLayer`: 은닉 계층을 위한 ReLU(Rectified Linear Unit) 활성화 함수와 그 미분을 구현합니다. 이 특정 구현은 음수 입력이 0.01로 스케일링되는 Leaky ReLU 변형을 사용합니다.
*   `StepFunction`: 네트워크의 출력(점수)을 이진 예측(0 또는 1)으로 변환하는 데 사용되는 간단한 계단 함수입니다.

### 네트워크 클래스 (`Net`)

`Net` 클래스는 신경망의 핵심 기능을 제공합니다:

*   `__init__(self, input, output)`: 입력 및 출력 차원과 선택적 은닉 계층으로 네트워크를 초기화합니다. 초기 가중치(`W`)와 편향(`b`)을 설정합니다.
*   `layer(self, node)`: 지정된 수의 노드를 가진 은닉 계층을 추가합니다.
*   `fix(self)`: `np.sqrt(2/self.node[i])`로 스케일링된 무작위 정규 분포를 사용하여 모든 계층의 가중치를 초기화합니다 (He 초기화).
*   `forward(self, inputArr)`: 네트워크를 통해 순전파를 수행하여 각 계층의 활성화를 계산합니다.
*   `loss(self, label)`: 역전파를 위한 손실(예측과 실제 레이블 간의 차이)을 계산합니다.
*   `backward(self, lr=1)`: 주어진 학습률(`lr`)로 경사 하강법을 사용하여 가중치와 편향을 업데이트하기 위한 역전파 알고리즘을 구현합니다.
*   `predict(self, x)`: 입력 배열 `x`에 대한 예측을 수행합니다.
*   `realPredict(self, t)`: 전체 손실을 평가하고 훈련 중에 가장 성능이 좋은 가중치와 편향을 추적합니다.

## 사용법

이 노트북은 다음을 수행하는 방법을 보여줍니다:

1.  **스파이럴 데이터 로드**: `load_spiral_data()`를 사용하여 이진 분류를 위한 2D 스파이럴 데이터셋을 생성합니다.
2.  **네트워크 초기화**: `Net` 클래스의 인스턴스를 생성하고, 은닉 계층 크기를 지정하고, 가중치를 초기화합니다.
3.  **네트워크 훈련**: 데이터셋을 반복하고, 순전파 및 역전파를 수행하며, 가중치와 편향을 업데이트합니다. 현재 예측 및 손실을 주기적으로 출력합니다.
4.  **결정 경계 시각화**: 훈련 후, 스파이럴 데이터셋에 대해 네트워크가 학습한 결정 경계를 플로팅합니다.

## 의존성

*   `numpy`
*   `matplotlib`

## 참고
http://www.gisdeveloper.co.kr/?p=8472
