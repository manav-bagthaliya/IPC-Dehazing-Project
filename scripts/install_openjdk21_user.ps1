<#
Simple user-level OpenJDK 21 installer (Adoptium Temurin)
- Downloads and extracts OpenJDK 21 to %LOCALAPPDATA%\Programs\OpenJDK\jdk-21
- Sets user-level `JAVA_HOME` and appends the JDK `bin` to the user `PATH` using `setx`.

This script avoids system (Machine) environment edits so it doesn't require Administrator.
Run with:
    powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\install_openjdk21_user.ps1"
#>

$ErrorActionPreference = 'Stop'

Write-Host "Starting user-level OpenJDK 21 install (Adoptium Temurin)..." -ForegroundColor Cyan

$installDir = Join-Path $env:LOCALAPPDATA "Programs\OpenJDK\jdk-21"
$zip = Join-Path $env:TEMP "OpenJDK21.zip"
$extrDir = Join-Path $env:TEMP "OpenJDK21Extract"

# Clean/create directories
if (Test-Path $zip) { Remove-Item $zip -Force -ErrorAction SilentlyContinue }
if (Test-Path $extrDir) { Remove-Item $extrDir -Recurse -Force -ErrorAction SilentlyContinue }
if (-not (Test-Path $installDir)) { New-Item -ItemType Directory -Path $installDir -Force | Out-Null }

$apiUrl = "https://api.adoptium.net/v3/binary/latest/21/ga/windows/x64/jdk/hotspot/normal"

Write-Host "Downloading OpenJDK 21..." -ForegroundColor Green
$downloadOk = $false
try {
    Invoke-WebRequest -Uri $apiUrl -OutFile $zip -UseBasicParsing -ErrorAction Stop
    $downloadOk = $true
}
catch {
    Write-Host "Invoke-WebRequest failed: $_. Falling back to WebClient..." -ForegroundColor Yellow
    try {
        $wc = New-Object System.Net.WebClient
        $wc.DownloadFile($apiUrl, $zip)
        $downloadOk = $true
    }
    catch {
        Write-Host "WebClient download failed: $_" -ForegroundColor Red
    }
}

if (-not $downloadOk) { throw "Download failed; cannot continue." }

Write-Host "Extracting archive..." -ForegroundColor Green
Expand-Archive -Path $zip -DestinationPath $extrDir -Force

$child = Get-ChildItem -Path $extrDir | Where-Object { $_.PSIsContainer } | Select-Object -First 1
if (-not $child) { throw "Extraction failed or no folder found in $extrDir" }

Write-Host "Copying JDK files to $installDir..." -ForegroundColor Green
Copy-Item -Path (Join-Path $child.FullName '*') -Destination $installDir -Recurse -Force

# Set user environment variables
$binPath = Join-Path $installDir "bin"
Write-Host "Setting user JAVA_HOME to $installDir" -ForegroundColor Green
setx JAVA_HOME "$installDir" | Out-Null

$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
if ($userPath -notlike "*$binPath*") {
    if ([string]::IsNullOrEmpty($userPath)) { $newPath = $binPath } else { $newPath = "$userPath;$binPath" }
    Write-Host "Adding $binPath to user PATH" -ForegroundColor Green
    setx Path "$newPath" | Out-Null
}
else {
    Write-Host "$binPath is already present in user PATH" -ForegroundColor Yellow
}

# Verify installation
$javaExe = Join-Path $binPath "java.exe"
if (Test-Path $javaExe) {
    Write-Host "Verifying installation by running: $javaExe -version" -ForegroundColor Cyan
    & "$javaExe" -version
}
else {
    Write-Host "ERROR: java.exe not found in $binPath" -ForegroundColor Red
}

# Cleanup
Remove-Item $zip -Force -ErrorAction SilentlyContinue
Remove-Item $extrDir -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "Installation complete. Close and reopen your terminal windows to pick up the new JAVA_HOME/PATH." -ForegroundColor Cyan
