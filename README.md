# Dehazing_terraform

## Dehazing-Transformer
- 건국대학교 컴퓨터공학부 졸업 작품
- 구성원: 남상대, 이태영, 장효진

## Data
- “AOD-Net: All-in-One Dehazing Network”논문에서 구축한 데이터셋을 사용
- 총 1449쌍의 원본 이미지와 hazy 이미지 데이터를 사용하였으며 6:3:1의 비율로 train/valid/test 데이터셋으로 구축
<p>
<img src="https://user-images.githubusercontent.com/61867199/138564574-f4ecb22b-bd14-4188-a8f3-e55bf42d2e60.jpg" width="300" height="200" >
<img src="https://user-images.githubusercontent.com/61867199/138564575-80728ded-6411-4460-8775-d98851200e80.jpg" width="300" height="200" >
</p>

## Model 
- IPT(Image Processing Transformer) 레퍼런스 모델 참고
<img src="https://user-images.githubusercontent.com/61867199/138564571-4bfecd05-f63d-43da-adc3-e3229e03fa4b.png" width="400" height="200">
- Knowledge Distillation 적용 예정
<img src="https://user-images.githubusercontent.com/61867199/138564568-94aae555-b158-432d-acb8-1cd6336e580f.png" width="400" height="200">

## Train
- optimizer: Adam
- learning rate: 1e-6
- Drop out: 0.2
- Metric: Mean Squared Error

## Product
- Application Framework: Streamlit
- CI/CD: Travis
- Model and Data are stored in Google Drive
