from ultralytics import YOLO

model = YOLO("yolo26n.pt")  # ← 여기만 yolov8n → yolo26n 으로 변경
results = model("https://ultralytics.com/images/bus.jpg", save=True)

print("감지 완료! runs/detect/predict 폴더 확인")