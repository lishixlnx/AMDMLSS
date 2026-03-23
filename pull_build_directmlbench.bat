@echo off
echo "Starting PAL-ML Clone"
git clone --depth 1 --branch main git@github.amd.com:PAL-Team/PAL-ML.git
pushd PAL-ML\apps\DirectMLBench
msbuild -p:RestorePackagesConfig=true -t:restore DirectMLBench.sln
msbuild DirectMLBench.sln
popd