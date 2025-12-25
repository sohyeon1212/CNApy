#!/bin/bash
#
# CNApy 실행 스크립트
# 사용법: ./run.sh [project_path] [scenario_path]
#

# 스크립트가 있는 디렉토리로 이동
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 가상환경 활성화 상태 확인 및 uv로 실행
echo "🧬 CNApy를 시작합니다..."
echo "   프로젝트 디렉토리: $SCRIPT_DIR"

# Qt 디버그 비활성화
export QT_LOGGING_RULES="*.debug=false"

# uv가 설치되어 있는지 확인
if command -v uv &> /dev/null; then
    echo "   uv를 사용하여 실행 중..."
    uv run python cnapy.py "$@"
else
    # uv가 없으면 가상환경 직접 사용
    if [ -d ".venv" ]; then
        echo "   가상환경(.venv)을 사용하여 실행 중..."
        source .venv/bin/activate
        python cnapy.py "$@"
    else
        echo "❌ 오류: uv가 설치되어 있지 않고, .venv 가상환경도 없습니다."
        echo "   먼저 'uv sync'를 실행하여 환경을 설정하세요."
        exit 1
    fi
fi

