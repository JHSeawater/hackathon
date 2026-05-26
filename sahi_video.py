from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import cv2
import json
from collections import Counter

# 모델 로드
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="yolo26n.pt",
    confidence_threshold=0.3,
    device="cpu",
)

# 영상 열기
INPUT_VIDEO = "people1.mp4"  # 자른 짧은 영상 파일명
OUTPUT_VIDEO = "people1_sahi_output.mp4"

cap = cv2.VideoCapture(INPUT_VIDEO)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 결과 영상 저장 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

all_frames = []
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # SAHI 슬라이스 추론
    result = get_sliced_prediction(
        frame,
        detection_model,
        slice_height=512,
        slice_width=512,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        verbose=0,  # 콘솔 출력 줄이기
    )

    # 박스 그리기
    detections = []
    for pred in result.object_prediction_list:
        x1, y1, x2, y2 = pred.bbox.to_xyxy()
        cls = pred.category.name
        conf = pred.score.value

        # 박스 + 라벨 그리기
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"{cls} {conf:.2f}", (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        detections.append({
            "class": cls,
            "confidence": round(conf, 2),
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
        })

    out.write(frame)

    # JSON 저장
    class_counts = Counter(d["class"] for d in detections)
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

    # 진행 상황 표시 (10프레임마다)
    if frame_idx % 10 == 0:
        print(f"진행: {frame_idx}/{total_frames} ({frame_idx*100//total_frames}%)")

    frame_idx += 1

cap.release()
out.release()

# JSON 저장
with open("detections_sahi.json", "w", encoding="utf-8") as f:
    json.dump(all_frames, f, ensure_ascii=False, indent=2)

print(f"\n완료! {OUTPUT_VIDEO}, detections_sahi.json 생성됨")