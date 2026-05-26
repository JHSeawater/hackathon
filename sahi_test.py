from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, get_prediction

# 모델 로드 (이미 받아둔 yolo26n.pt 재사용)
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="yolo26n.pt",
    confidence_threshold=0.3,
    device="cpu",
)

# [A] 일반 추론 (SAHI 없이)
print("=== 일반 추론 ===")
result_normal = get_prediction("parking.jpg", detection_model)
result_normal.export_visuals(export_dir="sahi_output/", hide_conf=False, file_name="normal")
print(f"감지된 객체: {len(result_normal.object_prediction_list)}개")

# [B] SAHI 슬라이스 추론
print("\n=== SAHI 슬라이스 추론 ===")
result_sliced = get_sliced_prediction(
    "parking.jpg",
    detection_model,
    slice_height=512,       # 슬라이스 크기 (드론 영상은 512가 무난)
    slice_width=512,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)
result_sliced.export_visuals(export_dir="sahi_output/", hide_conf=False, file_name="sliced")
print(f"감지된 객체: {len(result_sliced.object_prediction_list)}개")

print("\nsahi_output/ 폴더 확인하세요!")