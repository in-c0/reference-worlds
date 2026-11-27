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
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir '..')).Path
$WindowsRoot = [IO.Path]::GetFullPath($env:WINDIR).TrimEnd('\')
if ($RepoRoot.StartsWith($WindowsRoot + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
    $Recommended = Join-Path $env:USERPROFILE 'reference-worlds'
    throw @"
RefWorld is checked out under the Windows system directory:
  $RepoRoot

That location can be owned by Administrators and blocks normal-user Git/output writes.
Do not add System32 as a global Git safe.directory exception.

Clone a fresh user-owned checkout instead:
  cd $env:USERPROFILE
  git clone https://github.com/in-c0/reference-worlds.git
  cd reference-worlds

Then rerun this picker from there. Recommended path:
  $Recommended
"@
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

if ($ForceNative) {
    $NativeRunner = Join-Path $ScriptDir 'run-windows-native-smoke.py'
    if (-not (Test-Path $NativeRunner -PathType Leaf)) {
        throw "Native Python runner not found: $NativeRunner"
    }

    $pythonExe = $null
    $pythonPrefix = @()
    if (Get-Command 'py' -ErrorAction SilentlyContinue) {
        $oldPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = 'Continue'
            & py -3.11 -c "import sys; assert sys.version_info[:2] == (3,11)" *> $null
            if ($LASTEXITCODE -eq 0) {
                $pythonExe = 'py'
                $pythonPrefix = @('-3.11')
            }
        } finally {
            $ErrorActionPreference = $oldPreference
        }
    }
    if (-not $pythonExe -and (Get-Command 'python' -ErrorAction SilentlyContinue)) {
        $pythonExe = 'python'
    }
    if (-not $pythonExe) {
        throw 'Python 3.10-3.12 is required. Install Python 3.11 and rerun.'
    }

    $args = @($pythonPrefix) + @(
        $NativeRunner,
        '--reference',$Reference,
        '--model-size',$ModelSize,
        '--run-name',$RunName
    )
    if ($SkipTests) { $args += '--skip-tests' }

    $oldPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $pythonExe @args
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldPreference
    }
    exit $code
}

$Launcher = Join-Path $ScriptDir 'run-windows-smoke.ps1'
if (-not (Test-Path $Launcher -PathType Leaf)) {
    throw "Launcher not found: $Launcher"
}
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
