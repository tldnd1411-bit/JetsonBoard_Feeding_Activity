# 넙치 먹이활성도 산출 (저장된 영상 파일 입력)
# 지정한 경로의 영상 파일을 읽어서 → 좌상단에 먹이활성도 % 오버레이한 mp4 저장
# + 시간대별 CSV 로그 저장
#
# optical flow(flow_intensity) + 전경비율(fg_pixel_ratio) + YOLO 트래킹 기반
# "개체별 이동량"(total_mv_px, avg_mv_px_per_object_sec) 세 가지 특징을 합쳐
# 먹이활성도 점수를 계산합니다. movement_per_second_feeding.csv를 만드는 오프라인
# 분석 스크립트와 동일한 정의를 쓰므로 같은 방식으로 비교/시각화할 수 있습니다.
#   - total_mv_px               : SAMPLE_SEC 구간 동안 추적된 모든 개체의 이동거리 합(px)
#   - avg_mv_px_per_object_sec  : 개체 1마리, 1초당 평균 이동거리(px)
#
# 실시간 카메라가 아니라 저장된 영상 파일을 읽기 때문에:
#   - 카메라 장치 설정(CAMERA_INDEX 등) 대신 VIDEO_INPUT_PATH만 지정하면 됩니다.
#   - 시간 기준을 실제 벽시계(wall clock)가 아니라 "영상 자체의 재생 시간"
#     (frame_index / video_fps)으로 계산합니다. 그래야 처리 속도가 실시간보다
#     빠르거나 느려도 CSV의 time_sec가 영상 재생 시간과 정확히 일치합니다.
#   - 영상 끝(더 이상 읽을 프레임이 없음)에 도달하면 자동으로 종료됩니다.
#     중간에 멈추고 싶으면 Ctrl+C로 종료할 수 있습니다.
#
# 주의: YOLO 로딩/추론 부분은 ultralytics 패키지의 표준 YOLO().track() API를
# 기준으로 작성했습니다. 별도의 패키지/호출 방식을 쓰는 버전이라면
# open_yolo_model()과 MovementAccumulator.update() 안의 모델 호출 부분만
# 그에 맞게 바꿔주면 됩니다.

import os
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

print("라이브러리 로드 완료")


## ⚙️ 설정 — 여기만 수정하세요

VIDEO_INPUT_PATH = "C:/Users/tldnd/AppData/Local/CapCut/Videos/Jetson_feeding_result_sample.mp4"   # TODO: 분석할 영상 파일 경로로 수정

SAMPLE_SEC   = 3           # 몇 초 간격으로 먹이활성도 재계산 (오프라인 분석 스크립트와 동일 개념)
RESIZE_WIDTH = 640         # optical flow 특징 계산용 축소 해상도 (영상 저장은 원본 해상도 유지)

EMA_SPAN = 10              # 특징 값 평활화 스팬 (SAMPLE_SEC=3 기준 10샘플=30초)

# --- YOLO 이동량 추적 설정 ---
YOLO_MODEL_PATH = "best.pt"    # TODO: 학습된 넙치 탐지/추적용 가중치 경로로 수정
YOLO_CONF_THRESH = 0.25               # 이 confidence 이상만 유효 탐지로 채택
YOLO_IMGSZ = RESIZE_WIDTH             # 추론 해상도. optical flow와 같은 스케일을 써서
                                       # 두 특징의 px 단위가 서로 어긋나지 않게 함
YOLO_TRACK_EVERY_N_FRAMES = 1          # 저장된 영상 파일 처리라 실시간 제약이 없으므로
                                        # 기본값은 매 프레임(1). 처리 속도가 너무 느리면
                                        # 2~3으로 올려서 프레임을 건너뛰어도 됩니다.

FEEDING_SCORE_WEIGHTS = {
    "flow_intensity": 0.2,
    "fg_pixel_ratio": 0.2,
    "avg_mv_px_per_object_sec": 0.6,
}

PROGRESS_PRINT_EVERY_SEC = 30   # 진행 상황을 몇 초(영상 기준)마다 콘솔에 출력할지

OUTPUT_DIR = "video_feeding_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

_video_stem = Path(VIDEO_INPUT_PATH).stem
_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_PATH   = os.path.join(OUTPUT_DIR, f"feeding_log_{_video_stem}_{_ts}.csv")
VIDEO_PATH = os.path.join(OUTPUT_DIR, f"feeding_overlay_{_video_stem}_{_ts}.mp4")

print(f"입력 영상 경로 : {VIDEO_INPUT_PATH}")
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


# ── min/max 정규화 (오프라인 스크립트의 per-video 방식과 동일 개념) ──
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


## 🔧 영상 파일 열기

def open_video(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"입력 영상 파일을 찾을 수 없습니다: {path}")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {path}")
    return cap


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
    cap = open_video(VIDEO_INPUT_PATH)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if not video_fps or video_fps <= 0:
        print("경고: 영상에서 FPS 정보를 읽지 못해 기본값 30.0을 사용합니다.")
        video_fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / video_fps if total_frames > 0 else None

    print(f"영상 해상도: {actual_w}x{actual_h}  FPS: {video_fps:.2f}  "
          f"총 프레임: {total_frames}  "
          f"길이: {duration_sec:.1f}sec" if duration_sec else
          f"영상 해상도: {actual_w}x{actual_h}  FPS: {video_fps:.2f}")

    print(f"YOLO 모델 로드 중: {YOLO_MODEL_PATH}")
    yolo_model = open_yolo_model(YOLO_MODEL_PATH)
    movement_acc = MovementAccumulator()
    frame_counter = 0

    writer = cv2.VideoWriter(
        VIDEO_PATH, cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (actual_w, actual_h)
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
    prev_sample_elapsed = None   # 마지막으로 특징을 계산했던 "영상 기준" 시각(sec)
    ema_flow = None
    ema_fg = None
    ema_move = None
    current_score_pct = 0.0
    current_object_count = 0
    last_progress_print = 0.0

    wall_start = time.monotonic()
    print("영상 기반 먹이활성도 산출 시작 (끝까지 자동 진행, Ctrl+C로 중단 가능)")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("영상 끝에 도달했습니다. 처리를 종료합니다.")
                break

            # 실제 벽시계 시간이 아니라, "영상 자체의 재생 시간" 기준으로 흐름을 계산.
            # (처리 속도가 실시간보다 빠르거나 느려도 CSV의 time_sec는 영상 재생 시간과 일치)
            elapsed = frame_counter / video_fps

            # --- YOLO 추적: N프레임마다 수행, SAMPLE_SEC 구간 동안 이동거리 누적 ---
            frame_counter += 1
            if frame_counter % YOLO_TRACK_EVERY_N_FRAMES == 0:
                infer_frame = preprocess_frame(frame, YOLO_IMGSZ)
                movement_acc.update(yolo_model, infer_frame, YOLO_CONF_THRESH, YOLO_IMGSZ)

            if prev_sample_elapsed is None or (elapsed - prev_sample_elapsed) >= SAMPLE_SEC:
                window_sec = SAMPLE_SEC if prev_sample_elapsed is not None else elapsed
                if window_sec <= 0:
                    window_sec = SAMPLE_SEC

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
                prev_sample_elapsed = elapsed

            draw_feeding_overlay(frame, current_score_pct, current_object_count)
            writer.write(frame)

            # 진행 상황 출력 (영상 기준 시간으로 PROGRESS_PRINT_EVERY_SEC마다)
            if elapsed - last_progress_print >= PROGRESS_PRINT_EVERY_SEC:
                last_progress_print = elapsed
                wall_elapsed = time.monotonic() - wall_start
                if duration_sec:
                    pct = min(100.0, elapsed / duration_sec * 100.0)
                    print(f"진행률 {pct:5.1f}%  (영상 {elapsed:.0f}/{duration_sec:.0f}sec, "
                          f"처리 경과 {wall_elapsed:.0f}sec, 현재 점수 {current_score_pct:.1f}%)")
                else:
                    print(f"영상 {elapsed:.0f}sec 처리, "
                          f"처리 경과 {wall_elapsed:.0f}sec, 현재 점수 {current_score_pct:.1f}%")

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
