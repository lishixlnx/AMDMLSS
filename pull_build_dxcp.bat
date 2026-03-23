@echo off
echo "Starting DXCP Clone"

set build_type=%1

git clone --depth 1 --branch amd/main git@github.amd.com:AMD-Radeon-Driver/drivers.git
pushd drivers
git submodule update --init drivers\d3d\dxcp

pushd drivers\d3d\dxcp
git switch amd/stg/dxcp
git submodule update --init --recursive

cmake -S . -B "builds/%build_type%" -G"Visual Studio 17 2022" -DDXCP_DEVELOPER_MODE=ON -DDXCP_DEBUG_PRINTS=ON -DDXCP_ENABLE_LOG_ERRORS=ON -DDXCP_ENABLE_LOG_WARNINGS=ON -DDXCP_ENABLE_PERF_TOOLS=ON -DCMAKE_BUILD_TYPE=%build_type%
cmake --build "builds/%build_type%" -- /maxcpucount:10