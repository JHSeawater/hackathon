from ultralytics import YOLO
from collections import Counter
import json

model = YOLO("yolo26n.pt")
results = model("people1.mp4", save=False, conf=0.3, stream=True)

# 프레임마다 감지 결과 저장할 리스트
all_frames = []

for frame_idx, result in enumerate(results):
    # 이 프레임에서 감지된 객체들
    detections = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]  # 예: "person", "car"
        confidence = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2] 박스 좌표

        detections.append({
            "class": class_name,
            "confidence": round(confidence, 2),
            "bbox": [round(v, 1) for v in xyxy]
        })

    # 클래스별 개수 카운트 (예: {"person": 12, "car": 3})
    class_counts = Counter(d["class"] for d in detections)

    # 위험도 산출 (예시 로직 - 시나리오에 맞게 수정 가능)
    person_count = class_counts.get("person", 0)
    if person_count >= 20:
        risk_level = "high"
    elif person_count >= 10:
        risk_level = "medium"
    else:
        risk_level = "low"

    all_frames.append({
        "frame": frame_idx,
        "counts": dict(class_counts),
        "risk_level": risk_level,
        "detections": detections
    })

# JSON 파일로 저장
with open("detections.json", "w", encoding="utf-8") as f:
    json.dump(all_frames, f, ensure_ascii=False, indent=2)

print(f"총 {len(all_frames)}개 프레임 분석 완료")
print(f"detections.json 파일 생성됨")