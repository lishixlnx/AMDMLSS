@REM Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
@echo off
"C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Tools\Llvm\x64\bin\clang-scan-deps.exe" %* 2>NUL
exit /b %ERRORLEVEL%
