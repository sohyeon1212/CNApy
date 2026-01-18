# CNApy TODO List

## ✅ Completed Features (2025-12-26)

### 1. ✅ Media Management (배지 관리)
- **파일**: `cnapy/gui_elements/media_management_dialog.py`
- **메뉴**: Scenario > Media Management (배지 관리)... (Ctrl+M)
- **기능**:
  - 박테리아용 배지 템플릿 (M9 Minimal, LB, Anaerobic M9)
  - 식물세포용 배지 템플릿 (MS Medium, Heterotrophic, Autotrophic)
  - 동물세포용 배지 템플릿 (DMEM, RPMI 1640, Serum-Free)
  - 사용자 정의 배지 저장/로드/적용
  - JSON 파일로 배지 내보내기/가져오기

### 2. ✅ Current Media Display (현재 배지 정보 표시)
- Media Management 다이얼로그 상단에 현재 교환 반응 상태 표시
- 활성화된 기질 uptake 및 산물 분비 정보 확인 가능

### 3. ✅ Dynamic FBA (동적 FBA)
- **파일**: `cnapy/gui_elements/dynamic_fba_dialog.py`
- **메뉴**: Analysis > Dynamic FBA (dFBA)...
- **기능**:
  - 시간에 따른 바이오매스 및 대사물질 농도 변화 시뮬레이션
  - Michaelis-Menten 흡수 동역학 지원
  - 다양한 ODE 솔버 선택 (RK45, BDF, LSODA 등)
  - 시뮬레이션 결과 그래프 및 CSV 내보내기
  - scipy.integrate.solve_ivp 기반 구현

### 4. ✅ Enhanced Flux Sampling (Flux Sampling 고도화)
- **파일**: `cnapy/flux_sampling.py`, `cnapy/gui_elements/flux_sampling_dialog.py`
- **메뉴**: Analysis > Flux Sampling...
- **기능**:
  - 기존 Random Sampling 유지
  - **새로운 Predicted Flux-Based Sampling 추가**:
    - 예측된 flux (FBA, MOMA 등) 결과를 기준으로 샘플링
    - min/max fraction으로 샘플링 범위 조절
    - Gaussian 노이즈 추가 옵션
    - 불확실성 정량화에 유용

### 5. ✅ E-Flux2 Method for Omics Integration
- **파일**: `cnapy/gui_elements/omics_integration_dialog.py`
- **메뉴**: Analysis > Omics Integration > Transcriptome-based Flux Prediction (LAD/E-Flux2)...
- **기능**:
  - 기존 LAD (Least Absolute Deviation) 방법 유지
  - **새로운 E-Flux2 방법 추가**:
    - Gene expression을 기반으로 반응 상한 제약
    - FBA 수행 후 L1-norm 최소화로 parsimonious 솔루션 계산
    - 정규화 옵션 및 최소 flux bound 설정
    - Reference: Kim & Lun (2016), PLOS Computational Biology

---

## Dependencies
- scipy (Dynamic FBA에 필요)
- matplotlib (Dynamic FBA 플롯에 필요)
- pandas, numpy (기존 의존성)

## Files Created/Modified
- `cnapy/gui_elements/media_management_dialog.py` (NEW)
- `cnapy/gui_elements/dynamic_fba_dialog.py` (NEW)
- `cnapy/gui_elements/omics_integration_dialog.py` (MODIFIED - E-Flux2 추가)
- `cnapy/gui_elements/flux_sampling_dialog.py` (MODIFIED - 고도화)
- `cnapy/flux_sampling.py` (MODIFIED - predicted flux-based sampling 추가)
- `cnapy/gui_elements/main_window.py` (MODIFIED - 메뉴 항목 추가)
