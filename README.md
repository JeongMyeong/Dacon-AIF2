# Dacon-AIF2

# AI프렌즈 시즌2 위성관측 활용 강수량 산출 대회
- 밝기온도, 지표타입, GMI와 DPR의 위도/경도를 활용하여 해당하는 구역의 강수량을 산출하는 문제.
    
# 데이터
- 2016년 ~ 2018년 관측된 자료 [train(76,345개) test(2,416개)]
- GPM Core 위성의 GMI/DPR 센서에서 북서태평양영역에서 관측된 자료.
- 특정 orbit에서 기록된 자료를 40X40 형태로 분할하여 제공.
- subset_######_@@.npy 파일로 제공.
- ######은 위성이 사용되기 시작한 이후로 몇 번째 지구를 돌았는지(orbit 번호)
- @@은 해당 orbit에서 몇 번째 subset인지 나타내는 수 orbit 별로 subset 갯수는 다름.
- 채널 0~8 : 밝기온도
- 채널9 : 지표 타입
    - 앞자리 0 : Ocean
    - 앞자리 1 : Coastal
    - 앞자리 3 : lnland water
- 채널10~13 : GMI경도, GMI위도, DPR경도, DPR위도

## Dacon에서 제공한 정보
- [GPM Core 위성의 데이터 수집 영상](https://www.youtube.com/watch?time_continue=89&v=eM78gFFxAII&feature=emb_title) - NASA 유튜브
- [영상 - GPM Core 위성 적용 사례 (Scanning a Snow Storm)](https://www.youtube.com/watch?v=SSKv4A_Cj5s&feature=emb_title) - NASA 유튜브

# 평가지표
- 평가지표에 주목해야한다. 
- MAE를 F1 Score로 나눈값으로 MAE는 실제 값이 0.1 이상에 대해서만 계산된다.
- F1 score는 0.1 이상인 값은 1, 0.1미만은 0이다.
- 실제값이 결측치(-9999.xxx) 인 경우 계산에서 제외된다.
- MAE는 0.1 이상 값에 대해서만 평가지표로 계산된다. --> MAE 0.1 이상의 값만 집중적으로 학습에 사용한다.
- F1 score는 전체 데이터에 대해 분류 문제로 접근한다. 이 때 평가지표와 같이 0.1 이상의 값들은 1, 미만인 값들은 0으로 두고 분류를 한다.
- MAE는 5fold 평균 1.145 정도 나온다.
    - 1.14887
    - 1.14137
    - 1.15049
    - 1.13537
    - 1.14576
- F1 score는 현재 5fold 중 1fold만 진행.
    - 0.79835

- 모든 test 데이터를 regression 예측하고 1과 0값에 대해 MAE를 넣으면??
- 1인 값은 그대로 test데이터를 넣고 0인값은 0.099999를 넣는다. --> 왜냐하면 분류가 1인 값은 그대로 MAE 계산하면 되는데 0인값은 실제값이 0.1 이상일 수 있으므로 0.1과 가장 가까운 0.099999를 넣음으로써 MAE 값을 최소화 시키기 위해서이다. --> 성능이 높아짐. 위 MAE, f1 score로 1.58106 점을 기록.

- feature 에 따라 MAE가 1.13xxxx 대로 떨어지는 것을 확인. score가 더 좋아질 것으로 기대 됨.
- target이 0 인 데이터를 활용할 방법을 찾아봐야한다. 0.1 이상인 데이터들로 학습한 뒤 0을 예측하고 이것에 약간의 noise를 주고 학습데이터로 추가해본다. 여기서 validation set에 대해서는 0인 데이터가 포함되어서는 안 된다. 0.1이상인 데이터로 학습한 모델로 0인 데이터를 예측하고 예측 값에 따라 noise의 크기도 조절 해보는 시도도 좋을것 같음.
- 주변 픽셀을 확인할 수 있게 데이터를 구성하여 CNN 으로 풀어보기...
- input 3x3 output 3x3 데이터로 만들어서 target이 0.1이상인 값이 하나라도 있는 것을 데이터로 만든다.
- optimizer 에서 polynomial 방식, cosine방식 그리고 warm up 을적용시켜 보기.
- 40x40xN 으로 학습한 결과와 0.1이상의 값들만 학습시킬때 앙상블 효과가 좋음.
- 40x40xN 으로 학습시킬 때 훈련셋에 0.1이상인 픽셀이 100개 이상인것을 90도 돌린 것으로 데이터를 증량시켰을 때 좋은 결과를 보임.
- [polynomial Decay](https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/keras/optimizer_v2/learning_rate_schedule.py#L267-L406)
    - step = min(step, decay_steps)
    - ((initial_learning_rate - end_learning_rate) *
              (1 - step / decay_steps) ^ (power)
             ) + end_learning_rate
    - decay_step 은 전체 iter 로 설정.
    - 훈련할때는 에폭을 decay_st 변수 설정할 때의 epochs+10 으로 해주었음. --> 최소 learning_rate에서 10번의 학습을 더 하기 위함.

```
batch_size=256
epochs=100
decay_st = (len(train)//batch_size+1)*epochs

poly_sche = tf.keras.optimizers.schedules.PolynomialDecay( 0.001, decay_st, end_learning_rate=1e-6, power=0.9)
opt = tf.keras.optimizers.Adam(poly_sche)


#tf.keras.optimizers.schedules.PolynomialDecay(
#    initial_learning_rate, decay_steps, end_learning_rate=0.0001, power=1.0,
#    cycle=False, name=None)


```
