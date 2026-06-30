# ============================================================================
# @file        build_and_test.ps1
# @brief       Build and run qtx unit tests on Windows (PowerShell).
# @author      QTX Project
# ============================================================================
#
# Modes:
#   .\build_and_test.ps1                 — release build + tests
#   .\build_and_test.ps1 -Mode debug     — debug build + tests
#   .\build_and_test.ps1 -Mode strict    — release + /WX (warnings as errors)
#   .\build_and_test.ps1 -Mode bench     — OOM Survival benchmark
#   .\build_and_test.ps1 -Mode all       — strict + release + bench
#
# Requirements:
#   - CMake 3.20+
#   - Ninja or Visual Studio 2022+ (MSVC v143+)
#   - C++23 support (MSVC 19.34+)
#
# Build architecture:
#   - HA-core: /W4, all warnings, C++23
#   - Bridge: relaxed warnings (interacts with ggml C-ABI)
#   - GGML shim: compiled as C++ on MSVC (C11 compatibility issues)
#   - Benchmark: separate executable
#
# ============================================================================

param(
    [ValidateSet("release", "debug", "strict", "bench", "all")]
    [string]$Mode = "release",

    [string]$Generator = "Ninja",

    [switch]$Verbose
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildDir = Join-Path $Root "build" $Mode

# ============================================================================
# === Helper functions ===
# ============================================================================

function Write-Step {
    param([string]$Message)
    Write-Host "`n=== $Message ===" -ForegroundColor Cyan
}

function Invoke-Build {
    param(
        [string]$BuildType,
        [bool]$StrictWarnings = $false
    )

    $cmakeBuildDir = Join-Path $Root "build" $Mode

    Write-Step "Configuring ($Mode, $BuildType)"

    $cmakeArgs = @(
        "-S", $Root,
        "-B", $cmakeBuildDir,
        "-G", $Generator,
        "-DCMAKE_BUILD_TYPE=$BuildType"
    )

    if ($StrictWarnings) {
        $cmakeArgs += "-DQTX_STRICT_WARNINGS=ON"
    }

    & cmake @cmakeArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "CMake configure FAILED" -ForegroundColor Red
        exit 1
    }

    Write-Step "Building ($Mode)"

    $buildArgs = @("--build", $cmakeBuildDir, "--config", $BuildType)
    if ($Verbose) {
        $buildArgs += "--verbose"
    }

    & cmake @buildArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Build FAILED" -ForegroundColor Red
        exit 1
    }

    Write-Host "Build OK: $cmakeBuildDir" -ForegroundColor Green
}

function Invoke-Tests {
    $testBin = Join-Path $BuildDir "qtx_tests.exe"
    if (-not (Test-Path $testBin)) {
        # Multi-config generators (VS) put binaries in subdirectories.
        $testBin = Join-Path $BuildDir "Release" "qtx_tests.exe"
        if (-not (Test-Path $testBin)) {
            $testBin = Join-Path $BuildDir "Debug" "qtx_tests.exe"
        }
    }

    if (-not (Test-Path $testBin)) {
        Write-Host "Test binary not found!" -ForegroundColor Red
        exit 1
    }

    Write-Step "Running tests"
    $testArgs = @()
    if ($Verbose) {
        $testArgs += "-v"
    }
    & $testBin @testArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Tests FAILED" -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Write-Host "All tests passed" -ForegroundColor Green
}

function Invoke-Benchmark {
    $benchBin = Join-Path $BuildDir "oom_survival.exe"
    if (-not (Test-Path $benchBin)) {
        $benchBin = Join-Path $BuildDir "Release" "oom_survival.exe"
    }

    if (-not (Test-Path $benchBin)) {
        Write-Host "Benchmark binary not found!" -ForegroundColor Red
        exit 1
    }

    Write-Step "Running OOM Survival Benchmark"
    & $benchBin
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Benchmark FAILED" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

# ============================================================================
# === Main dispatch ===
# ============================================================================

switch ($Mode) {
    "release" {
        Invoke-Build -BuildType "Release"
        Invoke-Tests
    }
    "debug" {
        Invoke-Build -BuildType "Debug"
        Invoke-Tests
    }
    "strict" {
        Invoke-Build -BuildType "Release" -StrictWarnings $true
        Invoke-Tests
    }
    "bench" {
        Invoke-Build -BuildType "Release"
        Invoke-Benchmark
    }
    "all" {
        Write-Step "Running all checks"
        foreach ($m in @("strict", "release", "bench")) {
            Write-Host "`n##### MODE: $m #####" -ForegroundColor Yellow
            & $MyInvocation.MyCommand.Path -Mode $m
            if ($LASTEXITCODE -ne 0) {
                Write-Host "FAILED in mode $m" -ForegroundColor Red
                exit 1
            }
        }
        Write-Host "`nAll checks passed!" -ForegroundColor Green
    }
}
