@echo off
setlocal enabledelayedexpansion

:: Default values
set COMPILER=clang
set BUILD_TYPE=release
set CLEANUP=0
set BUILD_SAMPLE_TESTS=0
set RUN_SAMPLE_TESTS=0
set BUILD_ALL_TESTS=0
set DEPLOY=
set DEPLOY_PATH=./amdmlss_redist

:: Skip past the :usage helper into the argument parser.
goto :parse_args

:: Function to display usage
:usage
echo Usage: %0 [-c^|--compiler ^<compiler^>] [-b^|--build ^<build_type^>] [options]
echo   -c, --compiler           Compiler: vs2022, vs2026, clang, all (default: clang)
echo   -b, --build              Build type: debug, release, all (default: release)
echo   --clean-up               Remove all build directories before building
echo   --build-sample-tests     Build sample tests but don't run them
echo   --run-sample-tests       Build and run sample tests (or just run if already built)
echo   --build-all-tests        Build sample tests, mlss-tester and unit tests
echo   -d, --deploy [path]      Deploy amdmlss and cmake files (default: ./amdmlss_redist)
echo.
echo   When using -c all, builds with all supported compilers.
echo   When using -b all, builds both debug and release configurations.
echo   Each build will be placed in build/^<compiler^>-^<build_type^> directory.
exit /b 1

:: Parse command line arguments
:parse_args
if "%~1"=="" goto :done_parsing

if /i "%~1"=="-c" goto :set_compiler
if /i "%~1"=="--compiler" goto :set_compiler
if /i "%~1"=="-b" goto :set_build
if /i "%~1"=="--build" goto :set_build
if /i "%~1"=="-h" goto :usage
if /i "%~1"=="--help" goto :usage
if /i "%~1"=="--clean-up" (
    set CLEANUP=1
    shift
    goto :parse_args
)
if /i "%~1"=="--build-sample-tests" (
    set BUILD_SAMPLE_TESTS=1
    shift
    goto :parse_args
)
if /i "%~1"=="--run-sample-tests" (
    set RUN_SAMPLE_TESTS=1
    shift
    goto :parse_args
)
if /i "%~1"=="--build-all-tests" (
    set BUILD_ALL_TESTS=1
    shift
    goto :parse_args
)
if /i "%~1"=="-d" goto :set_deploy
if /i "%~1"=="--deploy" goto :set_deploy

echo Unknown option: %~1
goto :usage

:set_compiler
shift
if "%~1"=="" (
    echo Error: -c/--compiler requires an argument
    goto :usage
)
set COMPILER=%~1
shift
goto :parse_args

:set_build
shift
if "%~1"=="" (
    echo Error: -b/--build requires an argument
    goto :usage
)
set BUILD_TYPE=%~1
shift
goto :parse_args

:set_deploy
set DEPLOY=1
shift
:: Check if next argument is a path (doesn't start with -)
if not "%~1"=="" (
    set FIRST_CHAR=%~1
    set FIRST_CHAR=!FIRST_CHAR:~0,1!
    if not "!FIRST_CHAR!"=="-" (
        set DEPLOY_PATH=%~1
        shift
    )
)
goto :parse_args

:done_parsing

:: HIP_PATH / ROCM_PATH must be supplied by the environment so the build
:: adapts to whatever ROCm install the user has. At least one of them must
:: be set; if only one is provided we mirror it to the other so downstream
:: CMake projects can pick whichever name they prefer.
if "%HIP_PATH%"=="" if "%ROCM_PATH%"=="" (
    echo ERROR: neither HIP_PATH nor ROCM_PATH is set in the environment.
    echo        Set one of them to your ROCm install root, for example:
    echo            set HIP_PATH=C:\opt\rocm
    echo            set ROCM_PATH=C:\opt\rocm
    exit /b 1
)
if "%HIP_PATH%"==""  set HIP_PATH=%ROCM_PATH%
if "%ROCM_PATH%"=="" set ROCM_PATH=%HIP_PATH%

:: Perform cleanup if requested
if %CLEANUP%==1 (
    echo Cleaning up build directories...
    if exist "build" (
        rmdir /s /q "build"
        echo Removed build directory
    )
    if exist "build-clang" (
        rmdir /s /q "build-clang"
        echo Removed build-clang directory
    )
    if exist "build_clang" (
        rmdir /s /q "build_clang"
        echo Removed build_clang directory
    )
    if exist "build-vs" (
        rmdir /s /q "build-vs"
        echo Removed build-vs directory
    )
    if exist "build_vs" (
        rmdir /s /q "build_vs"
        echo Removed build_vs directory
    )
    if exist "buildvsdebug" (
        rmdir /s /q "buildvsdebug"
        echo Removed buildvsdebug directory
    )
    echo Clean-up completed!
    echo.
)

:: Validate build type
if /i not "%BUILD_TYPE%"=="debug" if /i not "%BUILD_TYPE%"=="release" if /i not "%BUILD_TYPE%"=="all" (
    echo Invalid build type: %BUILD_TYPE%
    goto :usage
)

:: Determine which build types to build
if /i "%BUILD_TYPE%"=="all" (
    set BUILD_TYPES=debug release
    echo Building both debug and release configurations...
) else (
    set BUILD_TYPES=%BUILD_TYPE%
)

:: Handle "all" compiler option
if /i "%COMPILER%"=="all" (
    echo Building with all supported compilers...
    echo.
    
    :: Build all combinations
    for %%t in (%BUILD_TYPES%) do (
        :: Build with vs2026
        echo ========================================
        echo Building with vs2026-%%t
        echo ========================================
        set PRESET=vs2026-%%t
        call :build_single
        if !errorlevel! neq 0 (
            echo vs2026-%%t build failed!
            exit /b 1
        )
        echo.
        
        :: Build with vs2022
        echo ========================================
        echo Building with vs2022-%%t
        echo ========================================
        set PRESET=vs2022-%%t
        call :build_single
        if !errorlevel! neq 0 (
            echo vs2022-%%t build failed!
            exit /b 1
        )
        echo.
        
        :: Build with clang
        echo ========================================
        echo Building with clang-%%t
        echo ========================================
        set PRESET=clang-%%t
        set LAST_PRESET=clang-%%t
        call :build_single
        if !errorlevel! neq 0 (
            echo clang-%%t build failed!
            exit /b 1
        )
        echo.
    )
    
    echo All builds completed successfully!
    
    :: Deploy if requested (use the last built configuration)
    if "%DEPLOY%"=="1" (
        echo.
        echo ========================================
        echo Deploying amdmlss
        echo ========================================
        call :deploy_amdmlss !LAST_PRESET! "%DEPLOY_PATH%"
        if !errorlevel! neq 0 (
            echo Deployment failed!
            exit /b 1
        )
    )
    
    exit /b 0
)

:: Validate single compiler
if /i not "%COMPILER%"=="vs2022" if /i not "%COMPILER%"=="vs2026" if /i not "%COMPILER%"=="clang" (
    echo Invalid compiler: %COMPILER%
    goto :usage
)

:: Build with single compiler (possibly multiple build types)
set BUILD_TYPE_COUNT=0
for %%t in (%BUILD_TYPES%) do set /a BUILD_TYPE_COUNT+=1

if %BUILD_TYPE_COUNT% gtr 1 (
    echo Building with compiler: %COMPILER%
    echo.
)

set LAST_PRESET=
for %%t in (%BUILD_TYPES%) do (
    set PRESET=%COMPILER%-%%t
    
    if %BUILD_TYPE_COUNT% gtr 1 (
        echo ========================================
        echo Building with preset: !PRESET!
        echo ========================================
    ) else (
        echo Building with preset: !PRESET!
    )
    
    call :build_single
    if !errorlevel! neq 0 (
        echo Build failed!
        exit /b 1
    )
    
    set LAST_PRESET=!PRESET!
    
    if %BUILD_TYPE_COUNT% gtr 1 (
        echo.
    )
)

echo Build completed successfully!

:: Deploy if requested
if "%DEPLOY%"=="1" (
    echo.
    echo ========================================
    echo Deploying amdmlss
    echo ========================================
    call :deploy_amdmlss %LAST_PRESET% "%DEPLOY_PATH%"
    if %errorlevel% neq 0 (
        echo Deployment failed!
        exit /b 1
    )
)

:: If no arguments provided, suggest using -c all
if "%~1"=="" (
    echo.
    echo Tip: Use "%~nx0 -c all" to build with all supported compilers
)
exit /b 0

:: Function to deploy amdmlss
:deploy_amdmlss
set PRESET_ARG=%~1
set DEPLOY_PATH_ARG=%~2

:: Extract build type from preset name
for /f "tokens=2 delims=-" %%a in ("%PRESET_ARG%") do set BUILD_CONFIG=%%a

:: Capitalize first letter of build config
set FIRST_LETTER=%BUILD_CONFIG:~0,1%
set REST=%BUILD_CONFIG:~1%
for %%a in (A B C D E F G H I J K L M N O P Q R S T U V W X Y Z) do (
    if /i "!FIRST_LETTER!"=="%%a" set FIRST_LETTER=%%a
)
set BUILD_CONFIG_CAPITALIZED=%FIRST_LETTER%%REST%

echo Deploying amdmlss to: %DEPLOY_PATH_ARG%

:: Create deployment directory
if not exist "%DEPLOY_PATH_ARG%" mkdir "%DEPLOY_PATH_ARG%"

:: Install using CMake with the correct configuration
cmake --install "build/%PRESET_ARG%" --config %BUILD_CONFIG_CAPITALIZED% --prefix "%DEPLOY_PATH_ARG%"

if %errorlevel% neq 0 (
    echo Deployment failed for %PRESET_ARG%!
    exit /b 1
)

echo Deployment completed successfully!
exit /b 0

:: Function to build with a single preset
:build_single
:: Extract build type from preset name
for /f "tokens=2 delims=-" %%a in ("%PRESET%") do set BUILD_CONFIG=%%a

:: Capitalize first letter of build config
set FIRST_LETTER=%BUILD_CONFIG:~0,1%
set REST=%BUILD_CONFIG:~1%
for %%a in (A B C D E F G H I J K L M N O P Q R S T U V W X Y Z) do (
    if /i "!FIRST_LETTER!"=="%%a" set FIRST_LETTER=%%a
)
set BUILD_CONFIG_CAPITALIZED=%FIRST_LETTER%%REST%

:: Build mlss-tester BEFORE the main project so the correct config is installed
if %BUILD_ALL_TESTS%==1 goto :build_tester
goto :skip_tester

:build_tester
echo.
echo Building amd-mlss-tester library...

set TESTER_SRC=3rdparty\amd-mlss-tester
set TESTER_INSTALL=%TESTER_SRC%\install

:: Map main-project preset to the matching mlss-tester preset
set FIRST_TOKEN=%PRESET:~0,5%
set TESTER_PRESET=
if /i "%FIRST_TOKEN%"=="clang" (
    if /i "%BUILD_CONFIG%"=="debug" (
        set TESTER_PRESET=clang-lib-static-debug
    ) else (
        set TESTER_PRESET=clang-lib-static-release
    )
)
if /i "%PRESET:~0,6%"=="vs2022" set TESTER_PRESET=vs2022-lib-static
if /i "%PRESET:~0,6%"=="vs2026" set TESTER_PRESET=vs2026-lib-static

if "%TESTER_PRESET%"=="" (
    echo No matching mlss-tester preset for '%PRESET%'!
    exit /b 1
)

set TESTER_CONFIG_ARGS=--preset %TESTER_PRESET% -DCMAKE_INSTALL_PREFIX=%TESTER_INSTALL% -DMLSS_ENABLE_HIP=ON -DBUILD_APP=OFF -DBUILD_TESTING=OFF

if /i "%FIRST_TOKEN%"=="clang" (
    set TESTER_CONFIG_ARGS=%TESTER_CONFIG_ARGS% -DCMAKE_CXX_COMPILER=%HIP_PATH%/bin/clang++.exe "-DCMAKE_CXX_FLAGS=-Wno-unused-command-line-argument"
)

cmake %TESTER_CONFIG_ARGS% -S %TESTER_SRC%
if %errorlevel% neq 0 (
    echo amd-mlss-tester configuration failed!
    exit /b 1
)

cmake --build "%TESTER_SRC%\build\%TESTER_PRESET%" --config %BUILD_CONFIG_CAPITALIZED%
if %errorlevel% neq 0 (
    echo amd-mlss-tester build failed!
    exit /b 1
)

cmake --install "%TESTER_SRC%\build\%TESTER_PRESET%" --config %BUILD_CONFIG_CAPITALIZED%
if %errorlevel% neq 0 (
    echo amd-mlss-tester install failed!
    exit /b 1
)
echo amd-mlss-tester built and installed successfully!

:skip_tester

:: Configure with CMake preset
echo.
echo Configuring with CMake preset: %PRESET%...
if %BUILD_ALL_TESTS%==1 (
    cmake --preset %PRESET% -DBUILD_SAMPLES=ON
) else (
    cmake --preset %PRESET%
)

if %errorlevel% neq 0 (
    echo CMake configuration failed for %PRESET%!
    exit /b 1
)

:: Build the project (including unit tests when mlss-tester is available)
echo Building project...
cmake --build "build\%PRESET%" --config %BUILD_CONFIG_CAPITALIZED%

if %errorlevel% neq 0 (
    echo Build failed for %PRESET%!
    exit /b 1
)

echo %PRESET% build completed successfully!

if %BUILD_ALL_TESTS%==1 (
    echo All tests built successfully!
)

:: Build sample tests if requested
if %BUILD_SAMPLE_TESTS%==1 goto :build_samples
if %RUN_SAMPLE_TESTS%==1 goto :build_samples
goto :skip_samples

:build_samples
echo.
echo Reconfiguring with BUILD_SAMPLES=ON for %PRESET%...
cmake --preset %PRESET% -DBUILD_SAMPLES=ON

if %errorlevel% neq 0 (
    echo CMake reconfiguration failed for %PRESET%!
    exit /b 1
)

echo Building sample tests for %PRESET%...
cmake --build "build\%PRESET%" --config %BUILD_CONFIG_CAPITALIZED%

if %errorlevel% neq 0 (
    echo Sample tests build failed for %PRESET%!
    exit /b 1
)
echo Sample tests built successfully!

:skip_samples

:: Run sample tests if requested
if %RUN_SAMPLE_TESTS%==1 (
    echo.
    echo Running sample tests for %PRESET%...
    
    :: First try the base samples directory (for Ninja builds)
    set TEST_DIR=build\%PRESET%\samples
    set HAS_EXE_FILES=0
    
    if exist "!TEST_DIR!" (
        for %%f in (!TEST_DIR!\*.exe) do (
            if exist "%%f" set HAS_EXE_FILES=1
        )
    )
    
    :: If not found, try with build configuration subdirectory (for MSBuild)
    if !HAS_EXE_FILES!==0 (
        set TEST_DIR=build\%PRESET%\samples\%BUILD_CONFIG_CAPITALIZED%
        if not exist "!TEST_DIR!" (
            set TEST_DIR=build\%PRESET%\samples\Debug
        )
        if not exist "!TEST_DIR!" (
            set TEST_DIR=build\%PRESET%\samples\Release
        )
    )
    
    if exist "!TEST_DIR!" (
        set TEST_FAILED=0
        for %%f in (!TEST_DIR!\*.exe) do (
            echo Running test: %%~nf
            "%%f"
            if !errorlevel! neq 0 (
                echo Test %%~nf failed!
                set TEST_FAILED=1
            ) else (
                echo Test %%~nf passed!
            )
            echo.
        )
        if !TEST_FAILED! neq 0 (
            echo Some tests failed for %PRESET%!
            exit /b 1
        )
        echo All sample tests passed for %PRESET%!
    ) else (
        echo No test directory found at !TEST_DIR!
        echo Skipping sample tests for %PRESET%
    )
)

exit /b 0
