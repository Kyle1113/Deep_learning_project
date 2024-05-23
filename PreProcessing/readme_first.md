# Deep_learning_project
### [commit log]
Featrue_extraction
1. 사용 이유 : 입력으로 들어온 의류(상의 or 하의)와 유사한 의류를 찾기 위한 유사도 분석 과정에서 모든 데이터셋과 비교를 하는 건 효율이 떨어지므로 입력데이터의 '카테고리', '색깔' 속성을 도출해서 같은 속성을 갖는 의류들하고만 유사도 분석을 진행하고자 함. 이때 '카테고리', '색깔' 속성을 뽑아내기 위한 모델로써 이용  
2. 입력 데이터 : 아래 데이터 전처리 과정을 통해서 얻은 상의를 크롭한 데이터 일부 이용
- train 303개, test 33
3. 모델 구조 : base를 공통으로 지나고 '카테고리'와 '색깔' 속성을 각각 예측하는 multioutput 모델 
- base(VGG) + dense : dense layer를 쌓으면서 분기 or 마지막 출력 레이어서만 분기
- base(res_net) + dense : dense layer를 쌓으면서 분기 or 마지막 출력 레이어서만 분기
- 즉 4가지 모델을 비교
3. 성능 비교 : validation의 정확도 비교
- '카테고리' 예측은 vgg(dense layer를 쌓으면서 분기)와 res_net(dense layer를 쌓으면서 분기)에서 가장 높은 성능
- '색깔' 예측은 vgg(dense layer를 쌓으면서 분기)에서 가장 높은 성능
**그러므로 vgg(dense layer를 쌓으면서 분기) 모델을 사용하는게 합당함**
  
### [commit log]
Data 전처리
1. image와 json 폴더의 이미지 식별 번호가 다른 파일은 삭제.
2. resize.py 
DATA size가 달라, 800 x 800으로 resize
3. kill_onepiece.py
상의, 하의만 구분할 것이기 때문에 그 외의 데이터 삭제(원피스, 아우터, 등등)
4. crop_image.py
상의와 하의를 폴리곤 좌표를 이용해 crop, 400x400으로 resize.
5. category_prepro.py
top_categories = ["탑", "블라우스", "니트웨어", "셔츠", "티셔츠", "브라탑", "후드"]
bottom_categories = ["스커트", "청바지", "팬츠", "레깅스", "조거팬츠"]
각 category 별로 폴더 생성.
6. color.py
colors = ["화이트", "블랙", "베이지", "핑크", "스카이블루", "블루", "그레이", "네이비",
          "브라운", "레드", "라벤더", "실버", "옐로우", "카키", "와인", "퍼플", "민트",
          "오렌지", "그린", "골드", "네온"]
각 color 별로 폴더 생성.


## PipeLine
![KakaoTalk_20240509_234531807](https://github.com/Kyle1113/Deep_learning_project/assets/168116920/de49ba05-e954-4d32-b1c5-ab0483f51931)
![KakaoTalk_20240509_234531807_01](https://github.com/Kyle1113/Deep_learning_project/assets/168116920/4a6dffb8-d65d-4ce8-8f7a-81e881f6db93)


### Fashion parsing models in TensorFlow
https://github.com/minar09/Fashion-Clothing-Parsing/blob/master/README.md

### K-Fashion 이미지 | AI 허브 공개 데이터
https://github.com/K-COORD/K-Fashion

### tensorflow 공식 파인튜닝 가이드
https://www.tensorflow.org/tfmodels/vision/instance_segmentation

### MaskRCNN-Modanet-Fashion-Segmentation-and-Classification
https://github.com/zekeriyyaa/MaskRCNN-Modanet-Fashion-Segmentation-and-Classification/blob/main/preprocess/parseData.ipynb
- 우리가 가지고 있는 data와 가장 유사한 데이터

