@REM Maven Wrapper script for Windows
@REM Downloads Maven if not already cached, then runs it.

@echo off
setlocal

set "MAVEN_PROJECTBASEDIR=%~dp0"
set "WRAPPER_PROPERTIES=%MAVEN_PROJECTBASEDIR%.mvn\wrapper\maven-wrapper.properties"

if not exist "%WRAPPER_PROPERTIES%" (
    echo Error: Could not find %WRAPPER_PROPERTIES% >&2
    exit /b 1
)

for /f "tokens=1,* delims==" %%a in ('findstr "distributionUrl" "%WRAPPER_PROPERTIES%"') do set "DISTRIBUTION_URL=%%b"

for %%i in ("%DISTRIBUTION_URL%") do set "MAVEN_ZIP=%%~nxi"
for /f "tokens=3 delims=-" %%v in ("%MAVEN_ZIP%") do set "MAVEN_VERSION=%%v"
set "MAVEN_VERSION=%MAVEN_VERSION:-bin.zip=%"

set "MAVEN_HOME=%USERPROFILE%\.m2\wrapper\dists\apache-maven-%MAVEN_VERSION%"

if not exist "%MAVEN_HOME%\bin\mvn.cmd" (
    echo Downloading Maven %MAVEN_VERSION%...
    powershell -Command "Invoke-WebRequest -Uri '%DISTRIBUTION_URL%' -OutFile '%TEMP%\maven.zip'"
    powershell -Command "Expand-Archive -Path '%TEMP%\maven.zip' -DestinationPath '%TEMP%\maven-extract' -Force"
    mkdir "%MAVEN_HOME%" 2>nul
    xcopy "%TEMP%\maven-extract\apache-maven-%MAVEN_VERSION%\*" "%MAVEN_HOME%\" /s /e /q >nul
    del "%TEMP%\maven.zip"
    rmdir /s /q "%TEMP%\maven-extract"
    echo Maven %MAVEN_VERSION% installed to %MAVEN_HOME%
)

set "M2_HOME=%MAVEN_HOME%"
set "PATH=%MAVEN_HOME%\bin;%PATH%"

mvn %*
