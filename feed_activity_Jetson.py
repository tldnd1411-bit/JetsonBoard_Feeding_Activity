# 넙치 먹이활성도 실시간 산출 (Jetson + USB 카메라)
# USB 카메라 입력 → 좌상단에 먹이활성도 % 오버레이한 mp4 저장 + 시간대별 CSV 로그 저장
#
# 종료 조건: 모니터가 연결되어 있지 않으므로 키 입력 대신 Ctrl+C로 강제 종료한다.

import os
import csv
import time
from datetime import datetime

import cv2
import numpy as np

print("라이브러리 로드 완료")


## ⚙️ 설정 — 여기만 수정하세요

CAMERA_INDEX  = 0          # USB 카메라 장치 번호 (/dev/video0)
CAMERA_WIDTH  = 1920
CAMERA_HEIGHT = 1080
CAMERA_FPS    = 30

SAMPLE_SEC   = 3           # 몇 초 간격으로 먹이활성도 재계산 (오프라인 분석 스크립트와 동일 개념)
RESIZE_WIDTH = 640         # 특징 계산용 축소 해상도 (Jetson 연산 부담 감소, 영상 저장은 원본 해상도 유지)

EMA_SPAN = 10              # 특징 값 평활화 스팬 (SAMPLE_SEC=3 기준 10샘플=30초)

FEEDING_SCORE_WEIGHTS = {
    "flow_intensity": 0.50,
    "fg_pixel_ratio": 0.50,
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

def draw_feeding_overlay(frame, score_pct):
    text = f"Feeding Activity: {score_pct:5.1f}%"

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

    writer = cv2.VideoWriter(
        VIDEO_PATH, cv2.VideoWriter_fourcc(*"mp4v"), actual_fps, (actual_w, actual_h)
    )

    csv_file = open(CSV_PATH, "w", newline="", encoding="utf-8-sig")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["time_sec", "flow_intensity", "fg_pixel_ratio", "feeding_score_pct"])

    normalizer = RunningNormalizer()
    prev_gray = None
    prev_sample_time = None
    ema_flow = None
    ema_fg = None
    current_score_pct = 0.0

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

            if prev_sample_time is None or (now - prev_sample_time) >= SAMPLE_SEC:
                resized = preprocess_frame(frame, RESIZE_WIDTH)
                gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

                if prev_gray is not None:
                    feats = compute_pair_features(prev_gray, gray)

                    ema_flow = feats["flow_intensity"] if ema_flow is None \
                        else ema_update(ema_flow, feats["flow_intensity"], EMA_SPAN)
                    ema_fg = feats["fg_pixel_ratio"] if ema_fg is None \
                        else ema_update(ema_fg, feats["fg_pixel_ratio"], EMA_SPAN)

                    norm_flow = normalizer.update_and_normalize("flow_intensity", ema_flow)
                    norm_fg   = normalizer.update_and_normalize("fg_pixel_ratio", ema_fg)

                    score = (FEEDING_SCORE_WEIGHTS["flow_intensity"] * norm_flow +
                             FEEDING_SCORE_WEIGHTS["fg_pixel_ratio"] * norm_fg)
                    current_score_pct = score * 100.0

                    csv_writer.writerow([
                        f"{elapsed:.2f}",
                        f"{feats['flow_intensity']:.4f}", f"{feats['fg_pixel_ratio']:.4f}",
                        f"{current_score_pct:.2f}",
                    ])
                    csv_file.flush()

                prev_gray = gray
                prev_sample_time = now

            draw_feeding_overlay(frame, current_score_pct)
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
