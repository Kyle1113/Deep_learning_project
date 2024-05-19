# Deep_learning_project
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

