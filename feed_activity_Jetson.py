# 넙치 먹이활성도 실시간 산출 (Jetson + USB 카메라)
# USB 카메라 입력 → 좌상단에 먹이활성도 % 오버레이한 mp4 저장 + 시간대별 CSV 로그 저장
#
# 이 버전은 기존 optical flow(flow_intensity) + 전경비율(fg_pixel_ratio) 방식에,
# YOLO 트래킹 기반 "개체별 이동량" 특징을 추가했습니다. 오프라인 분석 스크립트가
# 만드는 movement_per_second_feeding.csv (total_mv_px, avg_mv_px_per_object_sec)와
# 같은 정의로 계산해서, 같은 로직/그래프 스크립트로 바로 비교/시각화할 수 있게 했습니다.
#   - total_mv_px               : SAMPLE_SEC 구간 동안 추적된 모든 개체의 이동거리 합(px)
#   - avg_mv_px_per_object_sec  : 개체 1마리, 1초당 평균 이동거리(px)
#
# 주의: YOLO 로딩/추론 부분은 ultralytics 패키지의 표준 YOLO().track() API를
# 기준으로 작성했습니다. "yolo26" 가중치가 이 API와 100% 동일하게 동작한다는
# 전제이며, 만약 별도의 패키지/호출 방식을 쓰는 버전이라면 open_yolo_model()과
# MovementAccumulator.update() 안의 모델 호출 부분만 그에 맞게 바꿔주면 됩니다.
#
# 종료 조건: 모니터가 연결되어 있지 않으므로 키 입력 대신 Ctrl+C로 강제 종료한다.

import os
import csv
import time
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

print("라이브러리 로드 완료")


## ⚙️ 설정 — 여기만 수정하세요

CAMERA_INDEX  = 0          # USB 카메라 장치 번호 (/dev/video0)
CAMERA_WIDTH  = 1920
CAMERA_HEIGHT = 1080
CAMERA_FPS    = 30

SAMPLE_SEC   = 3           # 몇 초 간격으로 먹이활성도 재계산 (오프라인 분석 스크립트와 동일 개념)
RESIZE_WIDTH = 640         # optical flow 특징 계산용 축소 해상도 (영상 저장은 원본 해상도 유지)

EMA_SPAN = 10              # 특징 값 평활화 스팬 (SAMPLE_SEC=3 기준 10샘플=30초)

# --- YOLO 이동량 추적 설정 ---
YOLO_MODEL_PATH = "best.pt"    # TODO: 학습된 넙치 탐지/추적용 가중치 경로로 수정
YOLO_CONF_THRESH = 0.25               # 이 confidence 이상만 유효 탐지로 채택
YOLO_IMGSZ = RESIZE_WIDTH             # 추론 해상도. optical flow와 같은 스케일을 써서
                                       # 두 특징의 px 단위가 서로 어긋나지 않게 함
YOLO_TRACK_EVERY_N_FRAMES = 2         # 매 프레임 추적하면 Jetson에 부담 -> N프레임마다 추적
                                       # (너무 크게 잡으면 track id 연속성이 끊어져
                                       #  이동거리가 과소/과대 추정될 수 있음. 2~3 권장)

FEEDING_SCORE_WEIGHTS = {
    "flow_intensity": 0.34,
    "fg_pixel_ratio": 0.33,
    "avg_mv_px_per_object_sec": 0.33,
}

OUTPUT_DIR = "realtime_feeding_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_PATH   = os.path.join(OUTPUT_DIR, f"feeding_log_{_ts}.csv")
VIDEO_PATH = os.path.join(OUTPUT_DIR, f"feeding_overlay_{_ts}.mp4")

print(f"CSV 저장 경로  : {CSV_PATH}")
print(f"영상 저장 경로 : {VIDEO_PATH}")


## 🔧 특징 계산 (오프라인 분석 스크립트의 flow_intensity / fg_pixel_ratio와 동일 로직)

def preprocess_frame(frame, resize_width):
    if resize_width is None or resize_width == -1:
        return frame
    h, w = frame.shape[:2]
    return cv2.resize(frame, (resize_width, int(h * resize_width / w)))


def compute_pair_features(prev_gray, curr_gray, motion_threshold=1.0):
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    mag    = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    moving = mag[mag > motion_threshold]
    flow_intensity = float(moving.mean()) if len(moving) > 0 else 0.0

    diff = cv2.absdiff(prev_gray, curr_gray)
    _, fg_mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg_pixel_ratio = float(fg_mask.mean() / 255.0)

    return {"flow_intensity": flow_intensity, "fg_pixel_ratio": fg_pixel_ratio}


def ema_update(prev_val, new_val, span):
    alpha = 2.0 / (span + 1)
    return alpha * new_val + (1 - alpha) * prev_val


# ── 실시간 min/max 정규화 (오프라인 스크립트의 per-video 방식과 동일 개념) ──
class RunningNormalizer:
    def __init__(self):
        self.min_val = {}
        self.max_val = {}

    def update_and_normalize(self, col, val):
        mn = min(self.min_val.get(col, val), val)
        mx = max(self.max_val.get(col, val), val)
        self.min_val[col], self.max_val[col] = mn, mx
        if mx > mn:
            return float(np.clip((val - mn) / (mx - mn), 0.0, 1.0))
        return 0.0


## 🔧 YOLO 기반 개체 이동량 추적
#
# model.track()으로 프레임마다 개체별 track id를 얻고, 같은 id가 직전 추적
# 프레임 대비 얼마나 이동했는지(px)를 SAMPLE_SEC 구간 동안 누적합니다.
# 구간이 끝나면:
#   total_mv_px              = 그 구간 동안 모든 track의 이동거리 합
#   avg_mv_px_per_object_sec = total_mv_px / (구간에 등장한 고유 개체 수) / (구간 길이, sec)
# (movement_per_second_feeding.csv의 total_mv_px / avg_mv_px_per_object_sec와 동일 정의)

def open_yolo_model(model_path):
    model = YOLO(model_path)
    return model


class MovementAccumulator:
    """SAMPLE_SEC 구간 동안 track별 이동거리를 누적하는 상태 저장 객체."""

    def __init__(self):
        self.prev_centers = {}   # track_id -> (cx, cy)  (직전 추적 프레임 기준, YOLO_IMGSZ 스케일)
        self.reset_window()

    def reset_window(self):
        self.total_mv_px = 0.0
        self.object_ids_in_window = set()

    def update(self, model, frame_for_infer, conf, imgsz):
        """frame_for_infer 한 장에 대해 추적을 수행하고, 이전 추적 프레임 대비 이동거리를 누적."""
        results = model.track(
            source=frame_for_infer, persist=True, verbose=False,
            conf=conf, imgsz=imgsz,
        )
        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return  # 이번 프레임엔 유효한 track이 없음 (탐지 없음 / id 미배정)

        ids = boxes.id.int().cpu().tolist()
        centers = boxes.xywh[:, :2].cpu().tolist()  # [[cx, cy], ...], infer_frame 좌표계

        current_ids = set()
        for track_id, (cx, cy) in zip(ids, centers):
            current_ids.add(track_id)
            self.object_ids_in_window.add(track_id)
            prev = self.prev_centers.get(track_id)
            if prev is not None:
                dist = float(np.hypot(cx - prev[0], cy - prev[1]))
                self.total_mv_px += dist
            self.prev_centers[track_id] = (cx, cy)

        # 이번 프레임에 더 이상 보이지 않는 track은 비교 기준에서 제거.
        # (나중에 같은 id로 다시 나타나도 새 시작점부터 이동거리를 다시 누적)
        for track_id in list(self.prev_centers.keys()):
            if track_id not in current_ids:
                del self.prev_centers[track_id]

    def consume_window(self, window_sec):
        """SAMPLE_SEC 구간이 끝났을 때 호출 — 누적값을 읽고 다음 구간을 위해 리셋."""
        object_count = len(self.object_ids_in_window)
        total_mv_px = self.total_mv_px
        if object_count > 0 and window_sec > 0:
            avg_mv_px_per_object_sec = total_mv_px / object_count / window_sec
        else:
            avg_mv_px_per_object_sec = 0.0
        self.reset_window()
        return total_mv_px, avg_mv_px_per_object_sec, object_count


## 🔧 USB 카메라 열기

def open_camera(index, width, height, fps):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        raise RuntimeError(f"카메라를 열 수 없습니다 (index={index})")
    return cap


## 🔧 카메라 실측 FPS 측정
# cap.get(cv2.CAP_PROP_FPS)는 드라이버가 "보고"하는 값이라 요청 FPS와 다르게
# 부정확한 경우가 많다 (특히 고해상도 MJPG). 실제로 뽑히는 속도를 짧게 측정해서
# VideoWriter에 넣지 않으면 저장된 영상이 실제보다 빠르게(배속) 재생된다.
def measure_actual_fps(cap, n_frames=30, warmup_frames=5):
    for _ in range(warmup_frames):
        cap.read()
    t0 = time.monotonic()
    count = 0
    for _ in range(n_frames):
        ret, _ = cap.read()
        if not ret:
            break
        count += 1
    elapsed = time.monotonic() - t0
    return count / elapsed if elapsed > 0 and count > 0 else None


## 🔧 오버레이 그리기

def draw_feeding_overlay(frame, score_pct, object_count=None):
    text = f"Feeding Activity: {score_pct:5.1f}%"
    if object_count is not None:
        text += f"  (obj: {object_count})"

    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    pad = 10
    cv2.rectangle(frame, (0, 0), (tw + pad * 2, th + pad * 2), (0, 0, 0), -1)
    cv2.putText(frame, text, (pad, th + pad), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 255), 2, cv2.LINE_AA)


## ▶ 메인 루프

def main():
    cap = open_camera(CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS)
    actual_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_fps = cap.get(cv2.CAP_PROP_FPS) or CAMERA_FPS
    actual_fps   = measure_actual_fps(cap) or reported_fps
    print(f"카메라 해상도: {actual_w}x{actual_h}  "
          f"보고된 FPS: {reported_fps:.1f}  실측 FPS: {actual_fps:.1f}")

    print(f"YOLO 모델 로드 중: {YOLO_MODEL_PATH}")
    yolo_model = open_yolo_model(YOLO_MODEL_PATH)
    movement_acc = MovementAccumulator()
    frame_counter = 0

    writer = cv2.VideoWriter(
        VIDEO_PATH, cv2.VideoWriter_fourcc(*"mp4v"), actual_fps, (actual_w, actual_h)
    )

    csv_file = open(CSV_PATH, "w", newline="", encoding="utf-8-sig")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "time_sec", "flow_intensity", "fg_pixel_ratio",
        "total_mv_px", "avg_mv_px_per_object_sec", "object_count",
        "feeding_score_pct",
    ])

    normalizer = RunningNormalizer()
    prev_gray = None
    prev_sample_time = None
    ema_flow = None
    ema_fg = None
    ema_move = None
    current_score_pct = 0.0
    current_object_count = 0

    start_time = time.monotonic()
    print("실시간 먹이활성도 산출 시작 (Ctrl+C로 종료 가능)")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("카메라 프레임을 읽을 수 없습니다. 종료합니다.")
                break

            now     = time.monotonic()
            elapsed = now - start_time

            # --- YOLO 추적: N프레임마다 수행, SAMPLE_SEC 구간 동안 이동거리 누적 ---
            frame_counter += 1
            if frame_counter % YOLO_TRACK_EVERY_N_FRAMES == 0:
                infer_frame = preprocess_frame(frame, YOLO_IMGSZ)
                movement_acc.update(yolo_model, infer_frame, YOLO_CONF_THRESH, YOLO_IMGSZ)

            if prev_sample_time is None or (now - prev_sample_time) >= SAMPLE_SEC:
                window_sec = SAMPLE_SEC if prev_sample_time is not None else (now - start_time)

                resized = preprocess_frame(frame, RESIZE_WIDTH)
                gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

                total_mv_px, avg_mv_px_per_object_sec, object_count = \
                    movement_acc.consume_window(window_sec)
                current_object_count = object_count

                if prev_gray is not None:
                    feats = compute_pair_features(prev_gray, gray)

                    ema_flow = feats["flow_intensity"] if ema_flow is None \
                        else ema_update(ema_flow, feats["flow_intensity"], EMA_SPAN)
                    ema_fg = feats["fg_pixel_ratio"] if ema_fg is None \
                        else ema_update(ema_fg, feats["fg_pixel_ratio"], EMA_SPAN)
                    ema_move = avg_mv_px_per_object_sec if ema_move is None \
                        else ema_update(ema_move, avg_mv_px_per_object_sec, EMA_SPAN)

                    norm_flow = normalizer.update_and_normalize("flow_intensity", ema_flow)
                    norm_fg   = normalizer.update_and_normalize("fg_pixel_ratio", ema_fg)
                    norm_move = normalizer.update_and_normalize(
                        "avg_mv_px_per_object_sec", ema_move
                    )

                    score = (FEEDING_SCORE_WEIGHTS["flow_intensity"] * norm_flow +
                             FEEDING_SCORE_WEIGHTS["fg_pixel_ratio"] * norm_fg +
                             FEEDING_SCORE_WEIGHTS["avg_mv_px_per_object_sec"] * norm_move)
                    current_score_pct = score * 100.0

                    csv_writer.writerow([
                        f"{elapsed:.2f}",
                        f"{feats['flow_intensity']:.4f}", f"{feats['fg_pixel_ratio']:.4f}",
                        f"{total_mv_px:.2f}", f"{avg_mv_px_per_object_sec:.2f}", object_count,
                        f"{current_score_pct:.2f}",
                    ])
                    csv_file.flush()

                prev_gray = gray
                prev_sample_time = now

            draw_feeding_overlay(frame, current_score_pct, current_object_count)
            writer.write(frame)

    except KeyboardInterrupt:
        print("사용자 중단 (Ctrl+C) → 종료")

    finally:
        cap.release()
        writer.release()
        csv_file.close()
        print(f"CSV 저장 완료 : {CSV_PATH}")
        print(f"영상 저장 완료: {VIDEO_PATH}")


if __name__ == "__main__":
    main()