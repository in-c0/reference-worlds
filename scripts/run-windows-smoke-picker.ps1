param(
    [ValidateSet('Auto','518','448','392','336','280')]
    [string]$ModelSize = 'Auto',

    [string]$RunName = (Get-Date -Format 'yyyyMMdd-HHmmss'),

    [switch]$SkipTests,
    [switch]$ForceNative
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LauncherName = if ($ForceNative) { 'run-windows-native-smoke.ps1' } else { 'run-windows-smoke.ps1' }
$Launcher = Join-Path $ScriptDir $LauncherName
if (-not (Test-Path $Launcher -PathType Leaf)) {
    throw "Launcher not found: $Launcher"
}

try {
    Add-Type -AssemblyName System.Windows.Forms
    [System.Windows.Forms.Application]::EnableVisualStyles()

    $dialog = New-Object System.Windows.Forms.OpenFileDialog
    $dialog.Title = 'Choose a RefWorld reference image'
    $dialog.Filter = 'Images (*.jpg;*.jpeg;*.png;*.webp;*.bmp;*.tif;*.tiff)|*.jpg;*.jpeg;*.png;*.webp;*.bmp;*.tif;*.tiff|All files (*.*)|*.*'
    $dialog.Multiselect = $false
    $pictures = [Environment]::GetFolderPath('MyPictures')
    if ($pictures -and (Test-Path $pictures)) {
        $dialog.InitialDirectory = $pictures
    } else {
        $dialog.InitialDirectory = $env:USERPROFILE
    }

    $result = $dialog.ShowDialog()
    if ($result -ne [System.Windows.Forms.DialogResult]::OK) {
        throw 'No reference image selected.'
    }
    $Reference = $dialog.FileName
} catch {
    Write-Warning "Windows file picker unavailable: $($_.Exception.Message)"
    $Reference = Read-Host 'Enter the full path to a reference image'
    if ([string]::IsNullOrWhiteSpace($Reference)) {
        throw 'A reference image is required.'
    }
}

Write-Host "Selected reference: $Reference" -ForegroundColor Cyan

$args = @(
    '-ExecutionPolicy','Bypass',
    '-File',$Launcher,
    '-Reference',$Reference,
    '-ModelSize',$ModelSize,
    '-RunName',$RunName
)
if ($SkipTests) { $args += '-SkipTests' }

& powershell @args
exit $LASTEXITCODE
