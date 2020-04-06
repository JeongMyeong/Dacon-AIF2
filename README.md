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
