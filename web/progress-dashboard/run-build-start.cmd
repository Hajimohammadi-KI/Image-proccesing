@echo off
setlocal
pushd "%~dp0"
echo Running Next.js build and start with execution policy bypass...
powershell -NoProfile -ExecutionPolicy Bypass -Command "npm run build; npm run start"
popd
endlocal
