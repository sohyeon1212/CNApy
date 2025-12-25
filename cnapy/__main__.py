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

import os
import sys
import site

# ============================================================================
# Qt 플러그인 경로 설정 (모든 Qt 관련 import 전에 실행되어야 함)
# ============================================================================
def _setup_qt_paths():
    """PyQt5의 Qt 플러그인 경로를 설정합니다."""
    try:
        import PyQt5
        pyqt5_path = os.path.dirname(PyQt5.__file__)
        qt5_plugins = os.path.join(pyqt5_path, 'Qt5', 'plugins')
        qt5_platforms = os.path.join(qt5_plugins, 'platforms')
        
        if os.path.exists(qt5_plugins):
            os.environ['QT_PLUGIN_PATH'] = qt5_plugins
        if os.path.exists(qt5_platforms):
            os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt5_platforms
            
        # QCoreApplication이 생성되기 전에 라이브러리 경로 설정
        from PyQt5.QtCore import QCoreApplication
        QCoreApplication.setLibraryPaths([qt5_plugins])
    except Exception:
        pass

_setup_qt_paths()
# ============================================================================

try:
    from jpype._jvmfinder import getDefaultJVMPath, JVMNotFoundException, JVMNotSupportedException

    try:
        getDefaultJVMPath()
    except (JVMNotFoundException, JVMNotSupportedException):
        for path in site.getsitepackages():
            # in one of these conda puts the JRE
            os.environ['JAVA_HOME'] = os.path.join(path, 'Library')
            try:
                getDefaultJVMPath()
                break
            except (JVMNotFoundException, JVMNotSupportedException):
                pass
except ImportError:
    pass

from cnapy.application import Application

def main_cnapy(
    project_path: None | str = None,
    scenario_path: None | str = None,
):
    Application(
        project_path=project_path,
        scenario_path=scenario_path,
    )

if __name__ == "__main__":
    main_cnapy()
