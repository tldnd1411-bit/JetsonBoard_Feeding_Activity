# 넙치 먹이활성도 실시간 산출

Jetson 환경에서 USB 카메라 영상을 입력받아 **넙치 먹이활성도(Feeding Activity)** 를 실시간으로 계산하고, 영상 좌상단에 활성도 값을 오버레이하여 MP4로 저장하는 프로그램,
동시에 시간대별 먹이활성도 계산 결과를 CSV 로그 파일로 저장

---

## 1. 주요 기능

- USB 카메라 실시간 입력 처리
- Optical Flow 기반 움직임 강도 계산
- 프레임 차분 기반 전경 픽셀 비율 계산
- EMA 기반 특징값 평활화
- 실시간 min/max 정규화를 통한 먹이활성도 점수 산출
- 영상 좌상단에 먹이활성도 퍼센트 오버레이
- MP4 영상 저장
- CSV 로그 저장
- 모니터가 없는 Jetson 환경을 고려하여 `Ctrl+C`로 종료

---

## 2. 실행 환경

### 하드웨어

- NVIDIA Jetson 보드
- USB 카메라
- 저장 공간이 충분한 로컬 디스크 또는 외부 저장장치

### 소프트웨어

- Python 3.x
- OpenCV
- NumPy

필요 패키지 예시:

```bash
pip install opencv-python numpy
```

Jetson 환경에서는 OpenCV가 카메라 및 비디오 인코딩을 제대로 지원하는지 확인 필요

```bash
python3 -c "import cv2; print(cv2.__version__)"
```

---

## 3. 파일 구조 예시

```text
project/
├── realtime_feeding_activity.py
├── README.md
└── realtime_feeding_output/
    ├── feeding_log_YYYYMMDD_HHMMSS.csv
    └── feeding_overlay_YYYYMMDD_HHMMSS.mp4
```

---

## 4. 설정값

코드 상단의 설정 영역에서 카메라 입력, 해상도, FPS, 샘플링 간격 등을 수정할 수 있음

```python
CAMERA_INDEX  = 0
CAMERA_WIDTH  = 1920
CAMERA_HEIGHT = 1080
CAMERA_FPS    = 30

SAMPLE_SEC   = 3
RESIZE_WIDTH = 640
EMA_SPAN = 10
```

### 주요 설정 설명

| 설정값 | 설명 |
|---|---|
| `CAMERA_INDEX` | USB 카메라 장치 번호입니다. 일반적으로 `/dev/video0`이면 `0`을 사용 |
| `CAMERA_WIDTH` | 카메라 입력 해상도 가로 크기 |
| `CAMERA_HEIGHT` | 카메라 입력 해상도 세로 크기 |
| `CAMERA_FPS` | 카메라 요청 FPS로, 실제 FPS와 다를 수 있음 |
| `SAMPLE_SEC` | 몇 초 간격으로 먹이활성도를 재계산할지 설정 |
| `RESIZE_WIDTH` | 특징 계산용 축소 해상도, 영상 저장은 원본 해상도로 유지됨 |
| `EMA_SPAN` | 특징값 평활화에 사용하는 EMA |

---

## 5. 먹이활성도 산출 방식

먹이활성도는 다음 두 가지 특징값을 기반으로 계산

### 5.1 Optical Flow 기반 움직임 강도

이전 샘플 프레임과 현재 샘플 프레임 사이의 움직임을 Farneback Optical Flow로 계산

```python
flow_intensity
```

움직임 벡터의 크기 중 일정 임계값 이상인 값들의 평균을 사용

### 5.2 프레임 차분 기반 전경 픽셀 비율

이전 샘플 프레임과 현재 샘플 프레임의 차이를 계산한 뒤, Otsu threshold를 적용하여 변화가 발생한 픽셀 비율을 계산

```python
fg_pixel_ratio
```

### 5.3 EMA 평활화

순간적인 노이즈를 줄이기 위해 각 특징값에 EMA를 적용

```python
ema_flow
ema_fg
```

### 5.4 실시간 정규화

실시간으로 관측된 최소값과 최대값을 기준으로 각 특징값을 0~1 범위로 정규화

```python
norm_flow
norm_fg
```

### 5.5 최종 점수 계산

현재 코드는 두 특징값을 동일한 비중으로 반영

```python
FEEDING_SCORE_WEIGHTS = {
    "flow_intensity": 0.50,
    "fg_pixel_ratio": 0.50,
}
```

최종 먹이활성도는 다음과 같이 계산

```text
feeding_score = flow_intensity_norm × 0.5 + fg_pixel_ratio_norm × 0.5
feeding_score_pct = feeding_score × 100
```

---

## 6. 출력 파일

프로그램 실행 시 `realtime_feeding_output` 폴더가 자동 생성되며, 실행 시각 기준으로 결과 파일이 저장

### CSV 로그

파일명 예시:

```text
feeding_log_20260707_091500.csv
```

CSV 컬럼:

| 컬럼명 | 설명 |
|---|---|
| `time_sec` | 프로그램 시작 후 경과 시간 |
| `flow_intensity` | Optical Flow 기반 움직임 강도 |
| `fg_pixel_ratio` | 프레임 차분 기반 전경 픽셀 비율 |
| `feeding_score_pct` | 최종 먹이활성도 점수 |

### MP4 영상

파일명 예시:

```text
feeding_overlay_20260707_091500.mp4
```

저장 영상에는 좌상단에 다음과 같은 텍스트가 표시

```text
Feeding Activity:  75.3%
```

---

## 7. 실행 방법

Python 파일이 있는 경로에서 다음 명령어를 실행

```bash
python3 feed_activity_Jetson.py
```

실행 후 다음과 같은 로그가 출력됩니다.

```text
라이브러리 로드 완료
CSV 저장 경로  : realtime_feeding_output/feeding_log_YYYYMMDD_HHMMSS.csv
영상 저장 경로 : realtime_feeding_output/feeding_overlay_YYYYMMDD_HHMMSS.mp4
카메라 해상도: 1920x1080  보고된 FPS: 30.0  실측 FPS: 29.8
실시간 먹이활성도 산출 시작 (Ctrl+C로 종료 가능)
```

---

## 8. 종료 방법

모니터가 연결되어 있지 않은 Jetson 환경을 고려하여 키 입력 종료는 사용하지 않음


```text
Ctrl + C
```

정상 종료 시 CSV와 영상 파일이 저장됨

```text
CSV 저장 완료 : realtime_feeding_output/feeding_log_YYYYMMDD_HHMMSS.csv
영상 저장 완료: realtime_feeding_output/feeding_overlay_YYYYMMDD_HHMMSS.mp4
```

---

## 9. 카메라 확인 방법

USB 카메라가 정상적으로 인식되었는지 확인

```bash
ls /dev/video*
```

예시:

```text
/dev/video0
```

카메라 장치 정보 확인:

```bash
v4l2-ctl --list-devices
```

지원 해상도 및 포맷 확인:

```bash
v4l2-ctl --device=/dev/video0 --list-formats-ext
```

---

## 10. 주의사항

### 10.1 실제 FPS와 요청 FPS가 다를 수 있음

코드에서는 `cap.get(cv2.CAP_PROP_FPS)` 값만 사용하지 않고, 짧은 시간 동안 실제 프레임을 읽어 실측 FPS를 계산

### 10.2 실시간 min/max 정규화 특성

프로그램 실행 초반에는 최소값과 최대값 범위가 충분히 쌓이지 않았기 때문에 먹이활성도 값이 불안정할 수 있음

따라서 초기 몇 샘플은 참고값으로 보고, 일정 시간이 지난 뒤의 값을 기준으로 해석하는 것이 좋음

---

