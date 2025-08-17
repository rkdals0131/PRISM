# PRISM - Point-cloud & RGB Integrated Sensing Module
## 포인트클라우드 & RGB 통합 센싱 모듈

고성능 ROS2 기반 LiDAR-카메라 융합 파이프라인으로, 실시간 포인트클라우드 인터폴레이션(32→96채널)과 다중 카메라 컬러 매핑을 통한 향상된 3D 인식을 제공합니다.

## 📋 주요 기능

- **실시간 LiDAR 업샘플링**: Catmull-Rom 스플라인 기반 32→96채널 인터폴레이션
- **다중 카메라 융합**: 듀얼 카메라 시스템의 색상 정보 통합
- **고성능 최적화**: SIMD 커널, OpenMP 병렬화, SOA 메모리 레이아웃
- **제로카피 메모리 관리**: 사전 할당된 메모리 풀을 통한 효율적 처리
- **FILC 스타일 그리드 인터폴레이션**: 1024×N 고정 그리드 기반 일관된 각도 샘플링

## 🚀 빠른 시작

### 설치 및 빌드

```bash
# ROS2 워크스페이스에서
cd /home/user1/ROS2_Workspace/ros2_ws
source .venv/bin/activate  # 가상환경 활성화

# 패키지 빌드
colcon build --packages-select prism

# 최적화 활성화 빌드
colcon build --packages-select prism --cmake-args -DPRISM_ENABLE_NATIVE_OPT=ON

# 환경 설정
source install/setup.bash
```

### 실행

```bash
# 메인 PRISM 파이프라인 실행
ros2 launch prism prism.launch.py

# 디버그 시각화 포함 실행
ros2 launch prism projection_debug.launch.py

# 커스텀 파라미터 파일 사용
ros2 launch prism prism.launch.py params_file:=/path/to/your_params.yaml
```

## 🏗️ 시스템 구조

### 노드 구성

#### 1. `prism_fusion_node` (메인 노드)
- **입력**:
  - LiDAR 포인트클라우드: `/ouster/points`
  - 카메라 1: `/usb_cam_1/image_raw`
  - 카메라 2: `/usb_cam_2/image_raw`
- **처리**:
  - 선택적 LiDAR 인터폴레이션 (FILC 스타일 그리드)
  - 3D 포인트를 카메라 이미지 평면에 투영
  - 각 카메라별 색상 추출
  - 다중 카메라 색상 융합
- **출력**:
  - 컬러 포인트클라우드: `/ouster/points/colored`
  - 인터폴레이션된 클라우드: `/prism/debug/interpolated` (선택적)

#### 2. `projection_debug_node` (디버그 노드)
- 카메라별 오버레이 이미지: `/prism/projection_debug/camera_1`, `/prism/projection_debug/camera_2`
- 통계 정보: `/prism/projection_debug/statistics`

### 핵심 컴포넌트

```
src/
├── core/                          # 코어 모듈
│   ├── calibration_manager.cpp   # 카메라 캘리브레이션 관리
│   ├── memory_pool.cpp           # 제로카피 메모리 풀 (🔧 메모리 누수 수정됨)
│   └── point_cloud_soa.cpp       # SIMD 최적화된 SOA 레이아웃
├── interpolation/                 # 인터폴레이션 엔진
│   ├── interpolation_engine.cpp  # 메인 인터폴레이션 로직
│   ├── catmull_rom_interpolator.cpp  # Catmull-Rom 스플라인
│   └── simd_kernels.cpp         # SIMD 벡터 연산
├── projection/                    # 투영 엔진
│   ├── projection_engine.cpp     # 3D→2D 투영
│   ├── color_extractor.cpp      # 이미지에서 색상 샘플링
│   └── multi_camera_fusion.cpp  # 다중 카메라 색상 융합
└── nodes/                         # ROS2 노드
    ├── prism_fusion_node.cpp     # 메인 융합 노드
    └── projection_debug_node.cpp # 디버그 시각화
```

## ⚙️ 설정

### 캘리브레이션 파일

CALICO 스타일 멀티 카메라 캘리브레이션:
- `config/multi_camera_intrinsic_calibration.yaml` - 카메라 내부 파라미터
- `config/multi_camera_extrinsic_calibration.yaml` - LiDAR-카메라 변환

### 주요 파라미터 (`config/prism_params.yaml`)

#### 토픽 설정
```yaml
topics:
  lidar_input: "/ouster/points"
  camera_1_input: "/usb_cam_1/image_raw"
  camera_2_input: "/usb_cam_2/image_raw"
  colored_output: "/ouster/points/colored"
  debug_interpolated: "/prism/debug/interpolated"
```

#### 동기화 설정
```yaml
synchronization:
  queue_size: 1              # 메시지 큐 크기
  min_interval_ms: 0         # 최소 처리 간격 (0=제한없음)
  use_latest_image: true     # 최신 이미지 캐시 사용
  image_freshness_ms: 150    # 이미지 신선도 체크
```

#### 인터폴레이션 설정
```yaml
interpolation:
  enabled: true              # 인터폴레이션 활성화
  scale_factor: 2.0          # 수직 빔 증폭 배율
  input_channels: 32         # 입력 LiDAR 채널 수
  grid_mode: true            # FILC 스타일 그리드 모드
  spline_tension: 0.5        # 스플라인 텐션
  discontinuity_threshold: 0.5  # 불연속 감지 임계값
```

#### 투영 설정
```yaml
projection:
  enable_distortion_correction: true  # 왜곡 보정
  enable_frustum_culling: true       # 시야각 외부 포인트 제거
  min_depth: 0.5                     # 최소 거리 (m)
  max_depth: 100.0                   # 최대 거리 (m)
  parallel_cameras: false             # 병렬 카메라 처리
  sample_stride: 2                   # 포인트 서브샘플링 (2=절반)
  max_points: 0                      # 최대 포인트 수 (0=무제한)
```

#### 색상 추출 설정
```yaml
extraction:
  enable_subpixel: true              # 서브픽셀 정밀도
  confidence_threshold: 0.7          # 신뢰도 임계값
  blur_kernel_size: 0                # 블러 커널 크기
  interpolation: "bilinear"          # 보간 방법 (nearest/bilinear/bicubic)
```

#### 융합 설정
```yaml
fusion:
  strategy: "weighted_average"       # 융합 전략
  confidence_threshold: 0.5          # 융합 신뢰도 임계값
  distance_weight_factor: 1.0        # 거리 가중치
  enable_outlier_rejection: true     # 이상치 제거
```

## 🔧 성능 최적화

### FILC 스타일 그리드 인터폴레이션
- 3D 클라우드를 `1024 × input_channels` 고정 그리드로 구성
- 방위각 기반 최근접 이웃 매칭
- 불연속성 인식 수직 블렌딩
- `scale_factor`로 행 업스케일링 (예: 32 → 64 채널)

### 성능 튜닝 팁

#### 처리 속도 향상
```yaml
projection:
  sample_stride: 3-4         # 포인트 수 감소
  max_points: 80000-100000   # 프레임당 최대 포인트 제한
  parallel_cameras: true     # 병렬 처리 활성화

synchronization:
  min_interval_ms: 0         # 인위적 제한 제거
```

#### 메모리 최적화
- 사전 할당 메모리 풀 사용
- SOA(Structure of Arrays) 레이아웃으로 캐시 효율성 향상
- SIMD 커널로 벡터 연산 최적화

### 이론적 최대 처리율
```
최대 발행률 ≈ min(LiDAR_rate, 1000 / avg_processing_ms)
예: 처리시간 42ms → ~23.8 Hz 상한
    LiDAR 19 Hz → 출력 ~19 Hz
```

## 🐛 디버깅

### 런타임 통계 확인
```bash
# 통계 정보 확인
ros2 topic echo /prism/projection_debug/statistics --once

# 발행 속도 모니터링
ros2 topic hz /ouster/points/colored

# 카메라별 오버레이 시각화
ros2 run rqt_image_view rqt_image_view
# 토픽 선택: /prism/projection_debug/camera_1
```

### 테스트 실행
```bash
# 전체 테스트
colcon test --packages-select prism
colcon test-result --verbose

# 개별 유닛 테스트
./build/prism/test_interpolation_engine
./build/prism/test_projection_engine
./build/prism/test_calibration_manager
./build/prism/test_memory_pool
```

## 🔨 최근 수정사항

### v1.0.1 - 메모리 누수 수정
- **문제**: Grid mode에서 `PoolDeleter{nullptr}` 사용으로 매 프레임마다 ~1MB 메모리 누수
- **해결**: `PoolDeleter`가 `pool==nullptr`일 때 일반 `delete` 수행하도록 수정
- **결과**: 메모리 사용량 안정화, 시스템 정지 문제 해결

## 📦 의존성

- **ROS2 Humble**
- **PCL 1.10+** - 포인트클라우드 처리
- **OpenCV 4+** - 이미지 연산
- **Eigen3** - 선형대수
- **yaml-cpp** - 설정 파일 로딩
- **TBB** - 병렬 알고리즘

## 🎯 QoS 및 신뢰성

- 기본 설정: `SensorDataQoS` (keep_last(1), best_effort)로 최소 지연시간
- 신뢰성이 필요한 경우: QoS 브리지를 통한 재발행 고려

## 📄 라이선스

패키지 매니페스트 및 저장소 라이선스 파일 참조

---

### 문의 및 기여

이슈 발생 시 GitHub 이슈 트래커를 통해 보고해주세요.