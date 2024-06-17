# 폴더 구조(Repo Structure) yolov7
 ┣ cfg
 ┣ data
 ┃ ┣ train
 ┃ ┃ ┣ images # train 이미지
 ┃ ┃ ┃ ┃ 20230502_0105608460_3.jpg
 ┃ ┃ ┃ ┃ 20230509_0105780592_1.jpg
 ┃ ┃ ┃ ┗ 20230510_0105817364_1.jpg
 ┃ ┃ ┗ labels # train 레이블
 ┃ ┃   ┃ 20230502_0105608460_3.txt
 ┃ ┃   ┃ 20230509_0105780592_1.txt
 ┃ ┃   ┗ 20230510_0105817364_1.txt
 ┃ ┣ val
 ┃ ┃ ┃ images # validation 이미지
 ┃ ┃ ┃ ┃ 20230511_0105879118_3.jpg
 ┃ ┃ ┃ ┗ 20230517_0106020897_1.jpg
 ┃ ┃ ┗ labels # validation 레이블
 ┃ ┃   ┃ 20230511_0105879118_3.txt
 ┃ ┃   ┗ 20230517_0106020897_1.txt
 ┃ ┣ test
 ┃ ┃ ┣ images #test 이미지
 ┃ ┃ ┃ ┗ 20230504_0105704762_2.jpg
 ┃ ┃ ┗ labels #test 레이블
 ┃ ┃   ┗ 20230504_0105704762_2.jpg
 ┃ ┣ custom.yaml #하자 클래스
 ┃ ┗ hyp,scratch.p5.yaml #하이퍼 파라미터 값
 ┣ deploy
 ┣ models
 ┣ runs
 ┃ ┣ detect
 ┃ ┃ ┗ detect_0605 # 예측 실행 코드 내 설정 name
 ┃ ┃   ┣ labels
 ┃ ┃   ┃ ┗ 20230504_0105704762_2.txt # 모델 예측 좌표 결과
 ┃ ┃   ┗ 20230504_0105704762_2.jpg # 모델 예측 사진 결과
 ┃ ┗ train
 ┃   ┗ train_0614 # 학습 실행 코드 내 설정 name
 ┃     ┗ weights
 ┃       ┗ yolov7_best.pt # 모델
 ┣ scripts
 ┣ utils
 ┣ detect.py # 예측 코드
 ┣ train.py # 학습 코드
 ┣ test.py # 테스트 코드
 ┣ yolov7x.pt # 훈련 시 가중치 파일
 ┣ README.md
 ┗ requirements.txt # 패키지(라이브러리)설치


 #학습 실행 코드 : python train.py --img 640 640 --device 0,1 --batch-size 32 --epochs 450 --data data/custom.yaml --hyp data/hyp.scratch.p5.yaml --cfg cfg/training/yolov7x-custom.yaml --weights yolov7x.pt --name train_0614

 #테스트 실행 코드 : python test.py --img 640 --device 0,1 --batch-size 32 --task test --data data/custom.yaml --conf 0.001 --iou 0.6 --weights runs/train/train_0614/weights/yolov7_best.pt --save-txt --name test_0614

 #예측 실행 코드 : python detect.py --img 640 --device 0,1 --source data/test/images/20230504_0105704762_2.jpg --conf 0.5 --iou 0.6 --weights runs/train/train_0614/weights/yolov7_best.pt --save-txt --name detect_0605


