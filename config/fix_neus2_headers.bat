REM Force complete CMake rebuild with correct CUDA include order

echo ===== Forcing Complete CMake Rebuild =====

cd .\src\NeuS2

REM Step 1: NUCLEAR OPTION - Delete entire build directory
rmdir /s /q build
mkdir build
cd build

REM Step 2: Set CUDA environment variables (PATH already contains CUDA)
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8

REM Step 3: CRITICAL - Temporarily disable conda CUDA headers
echo Temporarily disabling conda CUDA headers...
ren "%CONDA_PREFIX%\include\crt" "crt_backup" 2>nul
ren "%CONDA_PREFIX%\include\cuda_runtime.h" "cuda_runtime.h.backup" 2>nul

REM Step 4: Activate VS2019 environment
REM Uncomment the following line if you're not already using the VS2019 Developer Command Prompt
REM call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"

echo ===== Clean CMake Configuration =====

REM Step 5: Set up CMake variables (use forward slashes for CMake compatibility)
REM Adjust paths as necessary for your vcpkg installation
set VCPKG_TOOLCHAIN=C:/repos/vcpkg/scripts/buildsystems/vcpkg.cmake
set VCPKG_TRIPLET=x64-windows
set CUDA_ROOT=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8
set CUDA_INCLUDE=%CUDA_ROOT%/include
set CUDA_COMPILER=%CUDA_ROOT%/bin/nvcc.exe

REM Step 6: Configure CMake from scratch
cmake .. -DCMAKE_TOOLCHAIN_FILE=%VCPKG_TOOLCHAIN% -DVCPKG_TARGET_TRIPLET=%VCPKG_TRIPLET% -DCUDA_TOOLKIT_ROOT_DIR="%CUDA_ROOT%" -DCUDA_INCLUDE_DIRS="%CUDA_INCLUDE%" -DCMAKE_CUDA_COMPILER="%CUDA_COMPILER%" -DCMAKE_BUILD_TYPE=RelWithDebInfo

if %ERRORLEVEL% NEQ 0 (
    echo CMake configuration failed
    goto RESTORE_HEADERS
)

echo ===== Building NeuS2 =====

REM Step 6: Build with clean configuration
cmake --build . --config RelWithDebInfo --parallel 4

set BUILD_RESULT=%ERRORLEVEL%

:RESTORE_HEADERS
echo ===== Restoring conda headers =====
ren "%CONDA_PREFIX%\include\crt_backup" "crt" 2>nul
ren "%CONDA_PREFIX%\include\cuda_runtime.h.backup" "cuda_runtime.h" 2>nul

if %BUILD_RESULT% EQU 0 (
    echo ===== SUCCESS! Testing NeuS2 =====
    cd ..
    python -c "import pyngp; print('NeuS2 imported successfully!')"
) else (
    echo ===== Build failed - Manual CMake edit needed =====
    echo.
    echo The include order is baked into CMakeLists.txt
    echo Need to manually edit the CMake configuration
)