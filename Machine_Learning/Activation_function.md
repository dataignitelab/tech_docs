# 활성화 함수 (Activation Function)

ANN(Arificial Neural Network)에서 레이어의 출력 값을 보정하기 위해서 사용하는 함수로서 목적에 따라 다양한 활성화 함수가 사용된다.

### 사용하는 이유?

먼저 선형성과 비선형성에 대한 개념를 짚고 넘어간다.

- 선형성(Linearity) : x와 y 의 관계가 고정된 비율에 따라 `비례 또는 반비례` 하는 성질
    - 그래프로 표현하면 직선의 형태를 띔
    - $y = wx+b$

![선형 그래프](/images/Machine_Learning/activation_function/0.png)

- 비선형성(Non-Linearity) : x 와 y 의 관계가 비례 하지 않는 성질
    - 그래프로 표현하면 곡선의 형태를 띔

![비선형 그래프](/images/Machine_Learning/activation_function/1.png)

실제 세상의 여러 문제들은 비선형성을 띄고 있는 것들이 대부분이다. 따라서 ANN 도 비선형 문제를 해결할수 있는 방안이 필요했다. 기본 뉴럴의 계산 로직은 선형함수( $y=WX+b$ ) 이다. 그래서 아무리 hidden layer를 많이 쌓더라도 결국 선형 연산( $f(a+b) = f(a) + f(b)$ )이기 때문에 비선형적인 문제를 해결할 수 없다. 아래 그림처럼 3번째 유형은 직선 하나만으로 분류할 수 없다.

![선형, 비선형 문제](/images/Machine_Learning/activation_function/2.png)

그에 대한 해결책이 바로 활성화 함수이다. 선형적인 ANN 레이어에 비선형성을 추가하는 것이다. 
활성화함수를 이용하여 선형연산의 연속이 아니라 비선형 연산의 연속으로 전환함으로서 비선형적인 복잡한 문제를 해결할 수 있도록 하는 것이 기본 개념이다. 아래 그림은 뉴런의 구조에 대한 내용으로 뉴런 노드(Cell Body)의 선형식 출력 대해서 $f$ (activation function)를 적용하여 출력으로 내보내는 것을 볼 수 있다.

![node](https://cs231n.github.io/assets/nn1/neuron_model.jpeg)

# 활성화 함수 종류
아래 설명하는 함수들 외에 다양한 활성화 함수들이 연구되고 있는 상황이며, 하나의 함수를 기반으로 발전된 형태가 주류를 이룬다. 한가지 명심해야할 것은 일부 초창기 활성화함수를 제외하고는 성능의 상하 관계가 있는 것이 아니라 용도에 따라 활성화 함수의 역할이 다른 것이니 이를 염두해두고 알맞게 사용하면 되겠다.


### **1. Step**

![step](/images/Machine_Learning/activation_function/3.png)

- 0 또는 1만 출력 하는 활성화 함수로 최초의 활성화 함수이다.
- 너무 단순해서 간단한 문제 외에는 적용하기 어려우며, 미분을 이용한 딥러닝 학습에 적용할 수 없다.

### **2. Sigmoid**

![Sigmoid](/images/Machine_Learning/activation_function/4.png)

- 0-1 사이로 만들어주는 딥러닝에서의 전통적인 활성화 함수로 데이터 평균을 0.5로 만들어준다. Nomailzation 역할을 수행한다고 보면 되겠다.
- **통계적 관점에서 Sigmoid**
    - Sigmoid 는 단순 0-1 사이의 숫자가 아니라 확률 값이다.
    - logistics 와 그리고 odds 와 probability 개념 이해
        - probability = 이벤트 발생할 확률
        - odds = 이벤트가 발생하지 하지 않을 확률 / 발생할 확률
        - logit : probability → odds 로 변환
        - logistics : odds → probability 로 변환, logit 의 역함수
    - 여기서 logistics 가  logistics regression 모델과 sigmoid 와 같은 생김새의 함수이다.
        - logistics regression : 회귀를 사용하여 데이터가 어떤 범주에 속할 확률을 0 ~ 1 사이로 예측하고 분류하는 알고리즘
    - 확률 관점에서 통계적, 전통적인 머신러닝에서 sigmoid 를 널리 사용했었고 딥러닝으로 확장되면서 자연스럽게 비선형성을 위하여 가장 널리 사용하던 sigmoid 가 활용되었다.
        - 딥러닝 관점에서는 비선형성을 추가하기 위한 목적이기 때문에 sigmoid 아니여도 된다. 하지만 딥러닝은 미분을 수행하므로 이왕이면 미분하기 쉬운, 연산이 간단해지는 함수를 활용하는 것이 좋으며 이에 가장 적합한것이 sigmoid 함수 였다.
        - 이에 연장선상에서 자연상수와 딥러닝과의 관계를 보자면 미분하기 가장 좋은 것이 자연상수이기 때문에 딥러닝과 관련된 대부분의 수식에서는 자연상수를 활용한 것들이 주류를 이룬다고 할 수 있다.
- 출력이 1 이하의 실수이기 때문에 다중 레이어에서 반복적으로 사용 시 기울기 소실(Vanising Gradient) 문제를 발생시켜서 학습이 되지 않는 문제가 있다.
- 그래프를 보면 알겠지만 0과 1로 데이터의 차이를 크게 만들어주는 역할을 수행하기 때문에 Binrary Classfication 문제에서 모델의 마지막 출력에만 적용되는 함수로 많이 활용된다.

### **3. tanh (Hyperbolic Tangent, 쌍곡 탄젠트)**

![tanh](/images/Machine_Learning/activation_function/5.png)

- Sigmoid와 거의 같으며 -1 ~ +1 사이로 만들어주는 활성화 함수로 데이터 평균을 0으로 만들어 준다.
- RNN 모델에서 주로 활용되는 함수이다. 이전 과거의 데이터를 현재 데이터에 반영해야 하므로 순환 연산을 수행하다보면 같은 값을 계속 곱하게 되는데,  0 ~ 1 사이의 값(sigmoid)이라면 기울기가 소실될 것이며, 1 보다 크다면 (ReLU) 기울기가 발산하게 되므로 적합하지 않다.
    - 그래서 적당한 타협점을 찾은 것이 tanh 이다.
    - Sigmoid 의 미분 최대값이 0.25이고, tanh 는 1이다. tanh 가 기울기가 소실될 가능성이 더 적다.

### **4. ReLU (Retified Linear Unit)**

![ReLU](/images/Machine_Learning/activation_function/7.png) 

- 음수는 0, 양수는 양수 그대로 출력하는 함수
- 초창기 ANN에서는 값의 Normalization 과 비선형성을 중시하여 Sigmoid 함수와 같은 것을 활용한 연구가 진행되다보니 Vanishing Gradient 문제를 해결하지 못하여 연구가 진행되지 못했다.
- ReLu 자체는 이미 오래전에 등장한 개념이었지만 실제 ANN에 적용하는 것은 그로부터 한참 뒤에 진행되었다. 선형 함수라고도 할수 있는 간단한 구조로 Vanising Gradient Problem을 해결하면서 적은 연산으로 깊은 레이어를 쌓으면서 안정적인 학습이 가능해짐에 따라 딥러닝이 발전하는 핵심적인 역할한 활성화 함수이다.
- 여기서 입력값이 음수가 된다는 것은 Relu에 의해 값이 0 된다는 것인데, 이 때 해당 값이 필요 없다는 의미보다 딥러닝 모델의 필요에 의해서 해당 값들은 0의 방향으로 학습한다는 관점 봐야한다.
- 거의 연산이 없기 때문에 학습 속도가 매우 빠르다. AlexNet 논문에 따르면 Sigmoid 에 비하여 6배 이상 빠르다고 한다.
    - **AlexNet**
        - ReLU를 ANN에 적용하여 딥러닝의 가능성을 보여준 딥러닝 모델이다.
        - 현재의 딥러닝 기술이 지금처럼 발전할 수 있도록 다양한 핵심 개념들(ReLU, Dropout, Auggmentation, Pooling, Nomailzation)을 정립하였다.
- 하지만 ReLu도 만능은 아니다. 아래 그림은 RNN 기반 모델에서 활성화 함수만 변경 했을때의 학습 결과이다. ReLu 에서 학습이 제대로 되지 않은 것을 볼 수 있다.
  - RNN 과 같이 같은 노드를 반복해서 계산이 일어나는 순환적 구조에서는 Sigmoid, tanh 가 더 좋은 성능을 보인다.
  - 그리고 모델의 레이어를 깊게 쌓지 않아서 Vanishing Gradient 문제를 일으키지 않을정도의 얇은 모델에서는 sigmoid 또는 tanh 가 ReLU 보다 좋은 성능을 낼 수도 있다.

![활성화함수에 따른 성능 차이 예시](/images/Machine_Learning/activation_function/6.png) 
            
### **5. ELU (Exponential Linear Unit)**

![ELU](/images/Machine_Learning/activation_function/8.png) 

- 음수일때 비선형적인 형태를 띔
- ReLU에서의 0 발생 시 학습이 되지 않는 Dying ReLU 문제가 발생하는 것을 해결하고자 나온 것으로 어느정도의 음수는 활용하자가 기본 개념이다. 이후 설명될 ReLU 의 모든 변형 함수들의 목적이다.
    - ReLu는 음수를 활용하지 않기 때문에 모델의 기울기 최저점에서 수렴하지 못하고 Loss가 요동치는 문제가 있는데, 음수를 활용하면 모델 수렴이 보다 잘 될 가능성이 있다.
    - 다만, ReLU보다 미분을 포함한 연산량이 늘어나면서 학습 시간이 더 많이 소요된다. 이는 다른 변형 함수들도 마찬가지다.
    
### **6. Leaky** **Relu**

![Leaky Relu](/images/Machine_Learning/activation_function/9.png) 

- 음수에는 0.01 을 곱하여 사용, 양수는 그대로 출력하는 함수
- ELU와 개발목적은 같다.
    
### **7. Swish**

![Swish](/images/Machine_Learning/activation_function/10.png) 

- ReLu 의 변형으로 작은 음수의 입력은 활용하는 함수이다.
- $f(x) = x * sigmoid(x)$

### **8. EXP (Exponential)**

![EXP](/images/Machine_Learning/activation_function/11.png) 

- 자연상수e의 지수함수이다. 활성화 함수의 범주에 넣기에는 애매한 측면이 있으나 빈도는 높지 않지만 주요 포인트들에서 활용되고 있으니 적어본다.
- 데이터 아주 작은 차이도 명확하게 구별 될 수 있는 큰 차이로 만들어주는 것이 특징이다.
- 회귀분석에서 모델 출력 값이 양수의 큰값이어야 할 때 활용될 수 있을 것으로 판단된다.
    - Yolo 같은 모델에서 오브젝트의 w,h 를 찾는데 활용된다.

### **9. Maxout**

![Maxout](/images/Machine_Learning/activation_function/12.png) 

- 활성화 함수를 구간 선형 함수(piecewise linear function)이라 가정하고, 뉴런 별 최적의 활성화 함수를 학습을 통하여 찾아내는 활성화 함수이다.
- 뉴런 별로 여러개의 선형 함수를 학습 시키도 최종적으로 최댓값들을 취하는 방식이다. 아래 그래프로는 빨간선이 되겠다.
- Maxout 도 결국 학습해야하므로 연산 파라미터가 증가하기 때문에 얼마만큼 활용할 것인가는 고민해봐야할 포인트며, Maxout 을 적용하면 신경망의 깊이를 줄일 수 있다는 장점 때문에 Maxout 에 의한 파라미터 증가를 어느정도 상쇄할 수 있다.

![Maxout](/images/Machine_Learning/activation_function/13.png) 


### **10. Softmax**

![softmax](https://blog.kakaocdn.net/dn/7o3ns/btqvQDIyhq4/FYgVfbO6NaJrkc7y11f440/img.png)

- 데이터의 합이 1인 확률 분포로 변환해주는 활성화 함수이다.
- 수식에 대한 부연 설명
  - k = 계산하고자하는 클래스, n = 전체 클래스 수
  - 단순하게 “k일 확률 / 전체 확률”
- 3개 이상의 다중 분류문제에서 모델의 마지막 레이어의 출력 값에 적용하는 활성화 함수로 많이 활용된다.
  - 클래스가 3개라면 각각의 클래스에 속할 확률을 3크기의 배열로 계산해준다.  
  - 일반적으로 Softmax를 적용하지 않은 모델의 아웃풋은 데이터의 크기가 정규화되어 있지 않아서 해당 값의 크기가 어느정도의 의미를 가지고 있는지 판단하기 어렵다. 따라서 Softmax를 적용하여 합이 1인 상태로 만들어서 상대적인 비교가 가능하도록 하는 것이 기본 목적이다.

# 참고자료

- 활성화 함수를 사용하는 이유에 대한 개념적 이해 : [https://www.inflearn.com/questions/486022](https://www.inflearn.com/questions/486022)
- Sigmoid, 자연상수e와 딥러닝과의 관계 : [https://www.facebook.com/groups/TensorFlowKR/posts/438895809784816/](https://www.facebook.com/groups/TensorFlowKR/posts/438895809784816/)
- 활성화 함수 종류 및 구현 : [https://gooopy.tistory.com/53](https://gooopy.tistory.com/53)
- 활성화 함수 종류 :  [https://velog.io/@cha-suyeon/DL-활성-함수activation-function](https://velog.io/@cha-suyeon/DL-%ED%99%9C%EC%84%B1-%ED%95%A8%EC%88%98activation-function)
- 딥러닝 강의 CS231n : [https://cs231n.github.io/neural-networks-1/?fbclid=IwAR2sIsBO5xiYRP783X5mMZkY-dxXeUFTw_A2BQw7eCHkyLTZ1H42p54nut8](https://cs231n.github.io/neural-networks-1/?fbclid=IwAR2sIsBO5xiYRP783X5mMZkY-dxXeUFTw_A2BQw7eCHkyLTZ1H42p54nut8)
- AlextNet 논문 : h[ttps://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
- RNN 에서 tanh 를 사용하는 이유 : [https://coding-yoon.tistory.com/132](https://coding-yoon.tistory.com/132)
