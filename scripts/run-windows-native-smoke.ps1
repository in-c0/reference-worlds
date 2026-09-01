param(
    [Parameter(Mandatory=$true)]
    [string]$Reference,

    [ValidateSet('Auto','518','448','392','336','280')]
    [string]$ModelSize = 'Auto',

    [string]$RunName = (Get-Date -Format 'yyyyMMdd-HHmmss'),

    [switch]$SkipTests
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir '..')).Path
Set-Location $RepoRoot

$VggtPin = 'a288dd0f14786c93483e45524328726ab7b1b4ce'
$VggtRepo = 'https://github.com/facebookresearch/vggt.git'

function Require-Command([string]$Name, [string]$Hint) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "$Name is required. $Hint"
    }
}

# Windows PowerShell 5.1 may turn a native process' stderr into ErrorRecords.
# Expected native failures/probes therefore run with non-terminating error handling
# and are judged only by the process exit code.
function Invoke-NativeProbe([string]$File, [string[]]$Arguments) {
    $oldPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $File @Arguments *> $null
        return $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldPreference
    }
}

function Invoke-NativeChecked([string]$Label, [string]$File, [string[]]$Arguments) {
    Write-Host $Label -ForegroundColor Cyan
    $oldPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $File @Arguments
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldPreference
    }
    if ($code -ne 0) {
        throw "$Label failed with exit code $code."
    }
}

function Invoke-PythonLogged([string]$Label, [string]$Python, [string[]]$Arguments, [string]$LogPath) {
    Write-Host $Label -ForegroundColor Cyan
    $oldPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $lines = & $Python @Arguments 2>&1 | Tee-Object -FilePath $LogPath -Append
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldPreference
    }
    return [pscustomobject]@{
        Code = $code
        Log = ($lines -join "`n")
    }
}

function Find-BasePython {
    $py = Get-Command 'py' -ErrorAction SilentlyContinue
    if ($py) {
        if ((Invoke-NativeProbe 'py' @('-3.11','-c','import sys; assert sys.version_info[:2] == (3,11)')) -eq 0) {
            return [pscustomobject]@{ Command='py'; Prefix=@('-3.11') }
        }
    }

    $python = Get-Command 'python' -ErrorAction SilentlyContinue
    if ($python) {
        if ((Invoke-NativeProbe 'python' @('-c','import sys; assert (3,10) <= sys.version_info[:2] < (3,13)')) -eq 0) {
            return [pscustomobject]@{ Command='python'; Prefix=@() }
        }
    }

    throw 'Python 3.10-3.12 is required. Install Python 3.11 from python.org, then rerun the same command.'
}

Require-Command 'nvidia-smi' 'Install/update the NVIDIA Windows driver.'
Require-Command 'git' 'Install Git for Windows.'

$ReferencePath = (Resolve-Path $Reference).Path
if (-not (Test-Path $ReferencePath -PathType Leaf)) {
    throw "Reference image not found: $ReferencePath"
}

Write-Host '== RefWorld native Windows GPU smoke ==' -ForegroundColor Cyan
Write-Host "Repository: $RepoRoot"
Write-Host "Reference:  $ReferencePath"

$GpuLine = (& nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
if (-not $GpuLine) { throw 'nvidia-smi returned no GPU.' }
$GpuParts = $GpuLine -split ','
$GpuName = $GpuParts[0].Trim()
$VramMiB = [int]$GpuParts[1].Trim()
Write-Host "GPU:        $GpuName ($VramMiB MiB VRAM)"

if ($ModelSize -eq 'Auto') {
    if ($VramMiB -ge 11800) { $ChosenSize = 518 }
    elseif ($VramMiB -ge 9800) { $ChosenSize = 448 }
    elseif ($VramMiB -ge 7600) { $ChosenSize = 392 }
    elseif ($VramMiB -ge 6000) { $ChosenSize = 336 }
    else { $ChosenSize = 280 }
} else {
    $ChosenSize = [int]$ModelSize
}
Write-Host "VGGT input: $ChosenSize x $ChosenSize"
if ($ChosenSize -ne 518) {
    Write-Warning 'Reduced resolution is a hardware smoke configuration, not the frozen benchmark baseline.'
}

$OutputRelative = "outputs/windows-smoke/$RunName"
$OutputHost = Join-Path $RepoRoot ($OutputRelative -replace '/', [IO.Path]::DirectorySeparatorChar)
New-Item -ItemType Directory -Force -Path $OutputHost | Out-Null
$LogPath = Join-Path $OutputHost "gpu-run-$ChosenSize.log"

$base = Find-BasePython
$VenvRoot = Join-Path $RepoRoot '.venv-refworld'
$VenvPython = Join-Path $VenvRoot 'Scripts\python.exe'

if (-not (Test-Path $VenvPython)) {
    Write-Host 'Creating repo-local Python environment (.venv-refworld)...'
    $venvArgs = @($base.Prefix) + @('-m','venv',$VenvRoot)
    Invoke-NativeChecked 'Creating Python virtual environment' $base.Command $venvArgs
}

Invoke-NativeChecked 'Bootstrapping pip/setuptools/wheel' $VenvPython @(
    '-m','pip','install','--disable-pip-version-check','--upgrade','pip','setuptools','wheel'
)

# Deliberately install the pinned wheels idempotently instead of probing a possibly
# absent torch import. This avoids the PowerShell 5.1 NativeCommandError footgun.
Invoke-NativeChecked 'Ensuring pinned PyTorch 2.3.1 CUDA 12.1 environment' $VenvPython @(
    '-m','pip','install',
    'torch==2.3.1','torchvision==0.18.1',
    '--index-url','https://download.pytorch.org/whl/cu121'
)

Invoke-NativeChecked 'Installing RefWorld/VGGT dependencies' $VenvPython @(
    '-m','pip','install','-e','.[dev,method]','huggingface_hub','einops','safetensors'
)

Invoke-NativeChecked 'Verifying CUDA from PyTorch' $VenvPython @(
    '-c',
    "import torch; print('torch', torch.__version__); print('cuda runtime', torch.version.cuda); print('gpu', torch.cuda.get_device_name(0)); assert torch.__version__.startswith('2.3.1'); assert torch.cuda.is_available()"
)

$UpstreamRoot = Join-Path $RepoRoot '.upstream'
$VggtRoot = Join-Path $UpstreamRoot 'vggt'
New-Item -ItemType Directory -Force -Path $UpstreamRoot | Out-Null

if (-not (Test-Path (Join-Path $VggtRoot '.git'))) {
    Invoke-NativeChecked 'Cloning pinned VGGT source' 'git' @('clone',$VggtRepo,$VggtRoot)
}

if ((Invoke-NativeProbe 'git' @('-C',$VggtRoot,'cat-file','-e',"$VggtPin^{commit}")) -ne 0) {
    Invoke-NativeChecked 'Fetching pinned VGGT commit' 'git' @('-C',$VggtRoot,'fetch','origin',$VggtPin)
}
Invoke-NativeChecked 'Checking out pinned VGGT commit' 'git' @('-C',$VggtRoot,'checkout','--detach',$VggtPin)
$head = (& git -C $VggtRoot rev-parse HEAD).Trim()
if ($head -ne $VggtPin) { throw "VGGT pin verification failed: $head" }

$env:PYTORCH_CUDA_ALLOC_CONF = 'max_split_size_mb:128'

if (-not $SkipTests) {
    $testResult = Invoke-PythonLogged 'Running focused contract tests...' $VenvPython @(
        '-m','pytest','-q',
        'tests/test_vggt_source.py','tests/test_source_geometry.py','tests/test_pinhole_warp.py',
        'tests/test_proposals.py','tests/test_splats.py','tests/test_schemas.py'
    ) $LogPath
    if ($testResult.Code -ne 0) {
        throw "Focused tests failed. See $LogPath"
    }
}

function Invoke-InferenceAttempt([int]$Size) {
    $SourceDir = Join-Path $OutputHost 'source-geometry'
    $SplatDir = Join-Path $OutputHost 'source-splat'
    $WarpDir = Join-Path $OutputHost 'warp-only'
    New-Item -ItemType Directory -Force -Path $SourceDir,$SplatDir,$WarpDir | Out-Null

    $result = Invoke-PythonLogged "Running REAL VGGT inference at ${Size}x${Size}..." $VenvPython @(
        '-m','refworld.runners.vggt_source',
        '--vggt-root',$VggtRoot,
        '--reference',$ReferencePath,
        '--output',$SourceDir,
        '--seed','0',
        '--model-size',[string]$Size
    ) $LogPath
    if ($result.Code -ne 0) { return $result }

    $result = Invoke-PythonLogged 'Building source-only 3DGS diagnostic...' $VenvPython @(
        '-m','refworld.runners.source_splat',
        '--reference',$ReferencePath,
        '--source-geometry',(Join-Path $SourceDir 'source-geometry.safe.json'),
        '--output',$SplatDir
    ) $LogPath
    if ($result.Code -ne 0) { return $result }

    $result = Invoke-PythonLogged 'Generating warp-only near-view neighborhood...' $VenvPython @(
        '-m','refworld.runners.warp_only',
        '--reference',$ReferencePath,
        '--source-geometry',(Join-Path $SourceDir 'source-geometry.safe.json'),
        '--output',$WarpDir
    ) $LogPath
    return $result
}

$result = Invoke-InferenceAttempt $ChosenSize
if ($result.Code -ne 0) {
    $looksLikeOom = $result.Log -match '(?i)(cuda.*out of memory|out of memory.*cuda|CUDNN_STATUS_ALLOC_FAILED)'
    if ($looksLikeOom -and $ChosenSize -gt 336) {
        $FallbackSize = 336
        Write-Warning "CUDA OOM at $ChosenSize. Retrying once at smoke-only ${FallbackSize}x${FallbackSize}."
        Remove-Item (Join-Path $OutputHost 'source-geometry') -Recurse -Force -ErrorAction SilentlyContinue
        Remove-Item (Join-Path $OutputHost 'source-splat') -Recurse -Force -ErrorAction SilentlyContinue
        Remove-Item (Join-Path $OutputHost 'warp-only') -Recurse -Force -ErrorAction SilentlyContinue
        $ChosenSize = $FallbackSize
        $LogPath = Join-Path $OutputHost "gpu-run-$ChosenSize.log"
        $result = Invoke-InferenceAttempt $ChosenSize
    }
}
if ($result.Code -ne 0) {
    throw "Inference failed. See $LogPath"
}

$ManifestPath = Join-Path $OutputHost 'source-geometry\source-geometry.safe.json'
if (-not (Test-Path $ManifestPath)) {
    throw 'Inference returned success but source-geometry manifest is missing.'
}
$Manifest = Get-Content $ManifestPath -Raw | ConvertFrom-Json
$PeakGiB = [math]::Round(([double]$Manifest.environment.peak_reserved_bytes / 1GB), 2)
$InferSec = [math]::Round([double]$Manifest.timing.inference_seconds, 2)
$LoadSec = [math]::Round([double]$Manifest.timing.model_load_seconds, 2)

Write-Host ''
Write-Host 'REAL INFERENCE COMPLETE' -ForegroundColor Green
Write-Host 'Execution:     native Windows CUDA'
Write-Host "GPU:           $($Manifest.environment.gpu_name)"
Write-Host "Model size:    $($Manifest.preprocessing.model_size)"
Write-Host "Model load:    $LoadSec s"
Write-Host "Inference:     $InferSec s"
Write-Host "Peak reserved: $PeakGiB GiB"
Write-Host "Output:        $OutputHost"
Write-Host "Manifest:      $ManifestPath"
Write-Host ''
Write-Host 'Artifacts include real VGGT depth/camera/confidence, a source-only 3DGS PLY, and warp-only near views.'
Write-Host 'Next: render source-splat/source-splat.ply at source-splat/source-camera.json and score it against the reference.'
