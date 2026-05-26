from ultralytics import YOLO

# 모델 로드 (이미 받아둔 파일 재사용됨)
model = YOLO("yolo26n.pt")

# 영상 분석 + 결과 영상 저장
results = model("people2.mp4", save=True, conf=0.25)

print("분석 완료! runs/detect/predict-N 폴더 확인")