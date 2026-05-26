import cv2
from ultralytics import YOLO
# 💡 [변경] 복잡한 내부 경로 대신 AutoDetectionModel 하나만 깔끔하게 가져옵니다.
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import visualize_object_predictions
from tqdm import tqdm

# 1. 환경 설정
video_path = "people2.mp4"
output_path = "people2_sahi_output.mp4"
model_path = "yolo26n.pt"

# 2. 💡 [변경] 가장 보편적이고 안전한 'from_model_type' 방식으로 다시 세팅하되,
# 혹시 모를 내부 대소문자 구분을 위해 model_type을 명확히 정의합니다.
detection_model = AutoDetectionModel.from_model_type(
    model_type="yolov8",      # Ultralytics 패키지는 'yolov8' 타입으로 통합 관리됩니다.
    model_path=model_path,
    confidence_threshold=0.25,
    device="cuda"             # 에러가 나면 "cpu"로 바꿔서 테스트해보세요.
)

# 3. 비디오 입출력 설정 (OpenCV)
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 저장을 위한 VideoWriter 설정 (MP4v 코덱 사용)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print(f"영상 분석 시작: 총 {total_frames} 프레임 진행 상황...")

# 4. 프레임별 루프 돌며 SAHI 적용
for _ in tqdm(range(total_frames)):
    ret, frame = cap.read()
    if not ret:
        break

    # OpenCV는 BGR 채널을 쓰지만, SAHI 내부 시각화를 위해 RGB로 변환하여 입력
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # SAHI 슬라이싱 추론 실행
    result = get_sliced_prediction(
        frame_rgb,
        detection_model,
        slice_height=512,       
        slice_width=512,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        verbose=0               
    )

    # SAHI 내장 함수로 프레임 위에 바운딩 박스 그리기
    visualized_result = visualize_object_predictions(
        image=frame,  
        object_prediction_list=result.object_prediction_list,
        rect_th=2,    
        text_size=0.5, 
        text_th=1     
    )

    # 결과 프레임을 새 영상 파일에 쓰기
    out.write(visualized_result["image"])

# 5. 자원 해제
cap.release()
out.release()
print(f"\n분석 완료! 저장된 파일: {output_path}")