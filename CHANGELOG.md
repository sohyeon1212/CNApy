# CNApy 변경 사항 (Changelog)

## 최근 변경 사항

이 문서는 CNApy에 추가된 주요 기능과 개선 사항을 기록합니다.

### 추가된 기능

#### 1. ROOM (Regulatory On/Off Minimization) 분석
- **위치**: Analysis 메뉴 > ROOM (Regulatory On/Off Minimization)
- **설명**: MILP solver를 사용하여 유전자 녹아웃 후 유의미한 플럭스 변화를 최소화하는 분석
- **요구사항**: MILP-capable solver (CPLEX, Gurobi, 또는 GLPK)
- **파일**: `cnapy/moma.py` (room 함수 추가)

#### 2. Flux Response Analysis
- **위치**: Analysis 메뉴 > Flux Response Analysis...
- **설명**: 타겟 반응의 플럭스를 스캔하면서 제품 반응의 최대 생산률을 플롯하는 분석
- **파일**: `cnapy/gui_elements/flux_response_dialog.py` (메뉴 연결 추가)

#### 3. Omics Integration (LAD Flux Prediction)
- **위치**: Analysis 메뉴 > Omics Integration > LAD Flux Prediction (transcriptome-based)...
- **설명**: Transcriptome 데이터를 기반으로 LAD (Least Absolute Deviation) 방법을 사용하여 플럭스 분포를 예측
- **기능**:
  - Gene expression 데이터 로드 (CSV, TSV, Excel)
  - Gene-to-reaction 매핑 (GPR rules 사용)
  - 다양한 aggregation 방법 지원 (min, max, mean, sum)
- **파일**: `cnapy/gui_elements/omics_integration_dialog.py` (신규 파일)

#### 4. 맵 기능 개선
- **PNG/SVG만으로 맵 생성**: JSON 파일 없이도 이미지 파일만으로 CNApy 맵을 생성할 수 있음
- **커스텀 반응 박스**: 모델에 없는 반응 ID도 맵에 박스로 추가하여 플럭스 값을 표시할 수 있음
- **파일**: `cnapy/gui_elements/central_widget.py`

#### 5. OptKnock 설명 개선
- **위치**: Strain Design 다이얼로그
- **개선 사항**: Inner/Outer Objective 설정에 대한 명확한 설명과 예시 추가
  - Outer Objective: 엔지니어가 최대화하고자 하는 목표 (예: EX_succ_e - 숙신산 생산)
  - Inner Objective: 세포가 최대화하는 목표 (예: BIOMASS - 성장)
- **파일**: `cnapy/gui_elements/strain_design_dialog.py`

### 수정된 파일 목록

- `cnapy/moma.py`: ROOM 함수 및 MILP solver 체크 함수 추가
- `cnapy/gui_elements/main_window.py`: ROOM, Flux Response Analysis, Omics Integration 메뉴 추가
- `cnapy/gui_elements/central_widget.py`: PNG-only 맵 생성 및 커스텀 반응 박스 기능 추가
- `cnapy/gui_elements/strain_design_dialog.py`: OptKnock 설명 개선
- `cnapy/gui_elements/flux_response_dialog.py`: 라이센스 헤더 추가
- `cnapy/gui_elements/omics_integration_dialog.py`: 신규 파일 (LAD 분석 다이얼로그)
- `cnapy/gui_elements/about_dialog.py`: 최근 추가 사항 표시
- `README.md`: 새로운 기능 설명 추가

### 라이센스

이 변경 사항들은 Apache License 2.0 하에 배포되며, 원본 CNApy 프로젝트의 일부입니다.

---

**참고**: 이 변경 사항들은 CNApy의 기능을 확장하고 개선하기 위해 추가되었습니다. 모든 변경 사항은 원본 CNApy 프로젝트의 라이센스와 호환됩니다.

