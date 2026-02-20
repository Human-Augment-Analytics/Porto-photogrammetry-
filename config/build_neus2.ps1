# NeuS2 Build Script - PowerShell Version
# Much cleaner than batch for this complexity

param(
    [string]$VcpkgRoot = "C:\repos\vcpkg",
    [string]$BuildType = "RelWithDebInfo",
    [string]$CondaEnv = "augenblick"
)

Write-Host "===== NeuS2 Build Script =====" -ForegroundColor Cyan

# Initialize conda for PowerShell
function Initialize-Conda {
    param([string]$EnvName)
    
    # Common conda installation paths
    $condaPaths = @(
        "$env:USERPROFILE\anaconda3",
        "$env:USERPROFILE\miniconda3", 
        "$env:USERPROFILE\mambaforge",
        "C:\ProgramData\anaconda3",
        "C:\ProgramData\miniconda3"
    )
    
    $condaPath = $null
    foreach ($path in $condaPaths) {
        if (Test-Path "$path\Scripts\conda.exe") {
            $condaPath = $path
            break
        }
    }
    
    if (!$condaPath) {
        throw "Conda installation not found! Please ensure conda is installed."
    }
    
    Write-Host "Found conda at: $condaPath" -ForegroundColor Green
    
    # Initialize conda for this PowerShell session
    & "$condaPath\Scripts\conda.exe" shell.powershell hook | Invoke-Expression
    
    # Activate the specified environment
    Write-Host "Activating conda environment: $EnvName" -ForegroundColor Yellow
    conda activate $EnvName
    
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to activate conda environment '$EnvName'. Does it exist?"
    }
    
    # Verify activation
    $activeEnv = $env:CONDA_DEFAULT_ENV
    if ($activeEnv -ne $EnvName) {
        throw "Environment activation failed. Active: $activeEnv, Expected: $EnvName"
    }
    
    Write-Host "Successfully activated environment: $activeEnv" -ForegroundColor Green
    return $env:CONDA_PREFIX
}

# Auto-detect CUDA architecture using proper CUDA tools
function Get-CudaArchitecture {
    try {
        # Use deviceQuery or nvcc to get actual compute capability
        $cudaPath = Find-CudaInstallation
        
        # Try using nvcc to compile a simple device query
        $tempDir = [System.IO.Path]::GetTempPath()
        $queryFile = Join-Path $tempDir "cuda_arch_query.cu"
        
        # Simple CUDA program to detect compute capability
        $cudaCode = @"
#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    printf("%d%d", prop.major, prop.minor);
    return 0;
}
"@
        
        # Write and compile the detection program
        Set-Content -Path $queryFile -Value $cudaCode
        $exePath = Join-Path $tempDir "cuda_arch_query.exe"
        
        & "$cudaPath\bin\nvcc.exe" $queryFile -o $exePath 2>$null
        
        if (Test-Path $exePath) {
            $result = & $exePath 2>$null
            Remove-Item $queryFile -ErrorAction SilentlyContinue
            Remove-Item $exePath -ErrorAction SilentlyContinue
            
            if ($result -match "^\d+$") {
                Write-Host "Detected CUDA compute capability: $($result.Substring(0,1)).$($result.Substring(1))" -ForegroundColor Green
                return $result
            }
        }
        
        # Fallback: let CMake auto-detect
        Write-Host "Could not detect architecture, letting CMake auto-detect..." -ForegroundColor Yellow
        return $null
        
    } catch {
        Write-Host "Architecture detection failed, letting CMake auto-detect..." -ForegroundColor Yellow
        return $null
    }
}

# Detect CUDA installation
function Find-CudaInstallation {
    $cudaPaths = @(
        "${env:CUDA_PATH}",
        "${env:CUDA_HOME}",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
    )
    
    foreach ($path in $cudaPaths) {
        if ($path -and (Test-Path "$path\bin\nvcc.exe")) {
            Write-Host "Found CUDA at: $path" -ForegroundColor Green
            return $path
        }
    }
    
    throw "CUDA installation not found!"
}

try {
    # Navigate to NeuS2 directory from config folder
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $projectRoot = Split-Path -Parent $scriptDir
    $neus2Dir = Join-Path $projectRoot "src\NeuS2"
    
    if (!(Test-Path $neus2Dir)) {
        throw "NeuS2 directory not found at $neus2Dir! Expected structure: project\src\NeuS2\"
    }
    
    if (!(Test-Path (Join-Path $neus2Dir "CMakeLists.txt"))) {
        throw "CMakeLists.txt not found in $neus2Dir! Are you sure this is the NeuS2 directory?"
    }
    
    Write-Host "Found NeuS2 at: $neus2Dir" -ForegroundColor Green
    Set-Location $neus2Dir
    
    # Initialize and activate conda environment
    $condaPrefix = Initialize-Conda -EnvName $CondaEnv
    
    # Detect GPU architecture (or let CMake handle it)
    $cudaArch = Get-CudaArchitecture
    if ($cudaArch) {
        Write-Host "Using detected CUDA architecture: $cudaArch" -ForegroundColor Yellow
        $archArgs = "-DCMAKE_CUDA_ARCHITECTURES=$cudaArch"
    } else {
        Write-Host "Letting CMake auto-detect CUDA architecture..." -ForegroundColor Yellow  
        $archArgs = ""
    }
    
    # Find CUDA installation
    $cudaPath = Find-CudaInstallation
    
    # Check conda environment
    if (!$condaPrefix) {
        throw "Conda environment not properly activated!"
    }
    Write-Host "Using conda environment: $condaPrefix" -ForegroundColor Green
    
    # Temporarily disable conda CUDA headers
    $condaCrt = Join-Path $condaPrefix "include\crt"
    $condaCudaHeader = Join-Path $condaPrefix "include\cuda_runtime.h"
    
    if (Test-Path $condaCrt) {
        Write-Host "Temporarily disabling conda CUDA headers..." -ForegroundColor Yellow
        Rename-Item $condaCrt "${condaCrt}_backup" -ErrorAction SilentlyContinue
    }
    if (Test-Path $condaCudaHeader) {
        Rename-Item $condaCudaHeader "${condaCudaHeader}.backup" -ErrorAction SilentlyContinue  
    }
    
    # Clean build directory
    if (Test-Path "build") {
        Remove-Item "build" -Recurse -Force
    }
    New-Item -ItemType Directory -Name "build" | Out-Null
    Set-Location "build"
    
    # Configure with CMake
    Write-Host "Configuring with CMake..." -ForegroundColor Cyan
    $cmakeArgs = @(
        "..",
        "-DCMAKE_TOOLCHAIN_FILE=$VcpkgRoot/scripts/buildsystems/vcpkg.cmake",
        "-DVCPKG_TARGET_TRIPLET=x64-windows", 
        "-DCUDA_TOOLKIT_ROOT_DIR=$cudaPath",
        "-DCMAKE_BUILD_TYPE=$BuildType"
    )
    
    # Add architecture if detected
    if ($archArgs) {
        $cmakeArgs += $archArgs
    }
    
    & cmake @cmakeArgs
    if ($LASTEXITCODE -ne 0) { throw "CMake configuration failed!" }
    
    # Build
    Write-Host "Building NeuS2..." -ForegroundColor Cyan
    & cmake --build . --config $BuildType --parallel
    if ($LASTEXITCODE -ne 0) { throw "Build failed!" }
    
    Write-Host "===== BUILD SUCCESSFUL! =====" -ForegroundColor Green
    
    # Test import
    Set-Location ".."
    python -c "import pyngp; print('NeuS2 imported successfully!')"
    
} catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
    exit 1
} finally {
    # Restore conda headers
    if ($condaPrefix) {
        $condaCrt = Join-Path $condaPrefix "include\crt_backup"
        $condaCudaHeader = Join-Path $condaPrefix "include\cuda_runtime.h.backup"
        
        if (Test-Path $condaCrt) {
            Rename-Item $condaCrt (Join-Path $condaPrefix "include\crt") -ErrorAction SilentlyContinue
        }
        if (Test-Path $condaCudaHeader) {
            Rename-Item $condaCudaHeader (Join-Path $condaPrefix "include\cuda_runtime.h") -ErrorAction SilentlyContinue
        }
    }
}