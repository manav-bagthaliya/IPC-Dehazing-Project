<#
PowerShell script: install_openjdk21.ps1
- Downloads Temurin (Adoptium) OpenJDK 21 (Windows x64) using Adoptium API
- Extracts to %LOCALAPPDATA%\Programs\OpenJDK\jdk-21
- Sets user-level JAVA_HOME and adds the JDK bin to user PATH
- Verifies installation by running `java -version`

Run (PowerShell, non-admin):
    powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\install_openjdk21.ps1"

Run as Administrator if you want to set system-wide environment variables.
#>

$ErrorActionPreference = 'Stop'

Write-Host "Starting OpenJDK 21 installer (Adoptium Temurin)..." -ForegroundColor Cyan

$installDir = Join-Path $env:LOCALAPPDATA "Programs\OpenJDK\jdk-21"
$zip = Join-Path $env:TEMP "OpenJDK21.zip"
$extrDir = Join-Path $env:TEMP "OpenJDK21Extract"

# Backup existing installation if present
if (Test-Path $installDir) {
    $timestamp = Get-Date -Format "yyyyMMddHHmmss"
    $backup = "$installDir-backup-$timestamp"
    Write-Host "Existing installation found. Backing up to: $backup"
    try {
        Rename-Item -Path $installDir -NewName "$installDir-backup-$timestamp" -ErrorAction Stop
    }
    catch {
        Write-Host "Unable to rename existing folder, attempting to remove it." -ForegroundColor Yellow
        Remove-Item -Path $installDir -Recurse -Force -ErrorAction SilentlyContinue
        New-Item -ItemType Directory -Path $installDir | Out-Null
    }
}
else {
    New-Item -ItemType Directory -Path $installDir -Force | Out-Null
}

# Clean any previous temp files
if (Test-Path $zip) { Remove-Item $zip -Force -ErrorAction SilentlyContinue }
if (Test-Path $extrDir) { Remove-Item $extrDir -Recurse -Force -ErrorAction SilentlyContinue }

# Adoptium API endpoint -- this will redirect to the actual binary
$apiUrl = "https://api.adoptium.net/v3/binary/latest/21/ga/windows/x64/jdk/hotspot/normal"

Write-Host "Downloading OpenJDK 21 from Adoptium..." -ForegroundColor Green
Invoke-WebRequest -Uri $apiUrl -OutFile $zip -UseBasicParsing

Write-Host "Extracting archive..." -ForegroundColor Green
Expand-Archive -Path $zip -DestinationPath $extrDir -Force

# The extracted content usually contains a single folder (e.g. jdk-21.X.X+...)
$child = Get-ChildItem -Path $extrDir | Where-Object { $_.PSIsContainer } | Select-Object -First 1
if (-not $child) { throw "Extraction failed or no folder found in $extrDir" }

# Move extracted contents into the install directory
Write-Host "Copying files to $installDir..." -ForegroundColor Green
Get-ChildItem -Path $child.FullName -Force | ForEach-Object {
    $dest = Join-Path $installDir $_.Name
    Move-Item -Path $_.FullName -Destination $dest -Force
}

# Determine if running as Administrator and set environment variables appropriately
$binPath = Join-Path $installDir "bin"
$windowsIdentity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($windowsIdentity)
$isAdmin = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if ($isAdmin) {
    Write-Host "Running as Administrator — setting system (Machine) environment variables." -ForegroundColor Green
    try {
        [Environment]::SetEnvironmentVariable("JAVA_HOME", $installDir, "Machine")
        $machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
        if ($machinePath -notlike "*$binPath*") {
            $newMachinePath = if ([string]::IsNullOrEmpty($machinePath)) { $binPath } else { "$machinePath;$binPath" }
            [Environment]::SetEnvironmentVariable("Path", $newMachinePath, "Machine")
            Write-Host "Added $binPath to system PATH." -ForegroundColor Green
        }
        else {
            Write-Host "$binPath is already present in system PATH" -ForegroundColor Yellow
        }
    }
    catch {
        Write-Host "Failed to set system environment variables: $_" -ForegroundColor Red
        Write-Host "Falling back to user environment variables." -ForegroundColor Yellow
        setx JAVA_HOME "$installDir" | Out-Null
        $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
        if ($userPath -notlike "*$binPath*") {
            $newPath = if ([string]::IsNullOrEmpty($userPath)) { $binPath } else { "$userPath;$binPath" }
            setx Path "$newPath" | Out-Null
        }
    }
}
else {
    Write-Host "Setting user JAVA_HOME to $installDir" -ForegroundColor Green
    setx JAVA_HOME "$installDir" | Out-Null

    # Update user PATH to include JDK bin
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($userPath -notlike "*$binPath*") {
        $newPath = if ([string]::IsNullOrEmpty($userPath)) { $binPath } else { "$userPath;$binPath" }
        Write-Host "Adding $binPath to user PATH" -ForegroundColor Green
        setx Path "$newPath" | Out-Null
    }
    else {
        Write-Host "$binPath is already present in user PATH" -ForegroundColor Yellow
    }
}

# Verify installation (call the new java executable directly)
$javaExe = Join-Path $binPath "java.exe"
if (Test-Path $javaExe) {
    Write-Host "Verifying installation by running: $javaExe -version" -ForegroundColor Cyan
    & "$javaExe" -version
}
else {
    Write-Host "ERROR: java.exe not found in $binPath" -ForegroundColor Red
}

# Cleanup temp files
Remove-Item $zip -Force -ErrorAction SilentlyContinue
Remove-Item $extrDir -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "Installation complete. Close and reopen your terminal windows to pick up the new JAVA_HOME/PATH." -ForegroundColor Cyan
