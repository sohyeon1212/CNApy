#!/usr/bin/env python3
#
# Copyright 2022 CNApy organization
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Qt 플러그인 경로 설정 (Qt 초기화 전에 실행되어야 함)
import os
import sys

def setup_qt_plugin_path():
    """Qt 플러그인 경로를 설정합니다 (PyQt5 사용 시 필요)."""
    try:
        import PyQt5
        pyqt5_path = os.path.dirname(PyQt5.__file__)
        qt5_path = os.path.join(pyqt5_path, 'Qt5')
        qt_plugins = os.path.join(qt5_path, 'plugins')
        qt_platforms = os.path.join(qt_plugins, 'platforms')
        
        # 환경변수 설정
        if os.path.exists(qt_plugins):
            os.environ['QT_PLUGIN_PATH'] = qt_plugins
        if os.path.exists(qt_platforms):
            os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt_platforms
            
        # QtCore를 먼저 임포트하고 라이브러리 경로를 명시적으로 추가
        from PyQt5 import QtCore
        if os.path.exists(qt_plugins):
            QtCore.QCoreApplication.addLibraryPath(qt_plugins)
        if os.path.exists(qt5_path):
            QtCore.QCoreApplication.addLibraryPath(qt5_path)
    except ImportError:
        pass

setup_qt_plugin_path()

from cnapy.__main__ import main_cnapy
from sys import argv

main_cnapy(
    project_path=None if len(argv) < 2 else argv[1],
    scenario_path=None if len(argv) < 3 else argv[2],
)
