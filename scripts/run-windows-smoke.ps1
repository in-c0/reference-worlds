param(
    [Parameter(Mandatory=$true)]
    [string]$Reference,

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
Set-Location $RepoRoot

$VggtPin = 'a288dd0f14786c93483e45524328726ab7b1b4ce'
$VggtRepo = 'https://github.com/facebookresearch/vggt.git'

function Require-Command([string]$Name, [string]$Hint) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "$Name is required. $Hint"
    }
}

Require-Command 'nvidia-smi' 'Install/update the NVIDIA Windows driver.'
Require-Command 'git' 'Install Git for Windows.'

$ReferencePath = (Resolve-Path $Reference).Path
if (-not (Test-Path $ReferencePath -PathType Leaf)) {
    throw "Reference image not found: $ReferencePath"
}

Write-Host '== RefWorld Windows GPU smoke ==' -ForegroundColor Cyan
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

function Show-Result([string]$ManifestPath, [int]$Size, [string]$Backend) {
    if (-not (Test-Path $ManifestPath)) {
        throw 'Inference returned success but source-geometry manifest is missing.'
    }
    $Manifest = Get-Content $ManifestPath -Raw | ConvertFrom-Json
    $PeakGiB = [math]::Round(([double]$Manifest.environment.peak_reserved_bytes / 1GB), 2)
    $InferSec = [math]::Round([double]$Manifest.timing.inference_seconds, 2)
    $LoadSec = [math]::Round([double]$Manifest.timing.model_load_seconds, 2)

    Write-Host ''
    Write-Host 'REAL INFERENCE COMPLETE' -ForegroundColor Green
    Write-Host "Execution:     $Backend"
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
}

function Invoke-DockerSmoke([int]$Size) {
    Write-Host 'Checking Docker GPU passthrough...'
    & docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
    if ($LASTEXITCODE -ne 0) {
        throw 'Docker is installed but cannot see the NVIDIA GPU. Use -ForceNative or fix Docker GPU passthrough.'
    }

    $HfCache = Join-Path $env:USERPROFILE '.cache\huggingface'
    New-Item -ItemType Directory -Force -Path $HfCache | Out-Null

    Write-Host 'Building pinned VGGT container...'
    & docker build -f docker/vggt.Dockerfile -t refworld-vggt .
    if ($LASTEXITCODE -ne 0) { throw 'Docker image build failed.' }

    $RunTests = if ($SkipTests) { '0' } else { '1' }
    $dockerArgs = @(
        'run','--rm','--gpus','all',
        '-e',"VGGT_MODEL_SIZE=$Size",
        '-e',"RUN_TESTS=$RunTests",
        '--mount',"type=bind,source=$ReferencePath,target=/data/reference.jpg,readonly",
        '--mount',"type=bind,source=$OutputHost,target=/workspace/reference-worlds/$OutputRelative",
        '--mount',"type=bind,source=$HfCache,target=/root/.cache/huggingface",
        'refworld-vggt',
        'bash','scripts/run-vggt-smoke.sh','/data/reference.jpg',$OutputRelative
    )

    Write-Host "Running real VGGT inference in Docker at ${Size}x${Size}..." -ForegroundColor Cyan
    $log = & docker @dockerArgs 2>&1 | Tee-Object -FilePath (Join-Path $OutputHost "gpu-run-$Size.log")
    return [pscustomobject]@{ Code=$LASTEXITCODE; Log=($log -join "`n") }
}

function Find-BasePython {
    $py = Get-Command 'py' -ErrorAction SilentlyContinue
    if ($py) {
        & py -3.11 -c "import sys; print(sys.executable)" *> $null
        if ($LASTEXITCODE -eq 0) {
            return [pscustomobject]@{ Command='py'; Prefix=@('-3.11') }
        }
    }
    $python = Get-Command 'python' -ErrorAction SilentlyContinue
    if ($python) {
        & python -c "import sys; assert sys.version_info >= (3,10) and sys.version_info < (3,13)" *> $null
        if ($LASTEXITCODE -eq 0) {
            return [pscustomobject]@{ Command='python'; Prefix=@() }
        }
    }
    throw 'Python 3.10-3.12 is required for native CUDA fallback. Install Python 3.11, then rerun this same command.'
}

function Invoke-NativeSetup {
    Write-Host 'Docker is unavailable/bypassed; using native Windows CUDA/PyTorch.' -ForegroundColor Yellow
    $base = Find-BasePython
    $VenvRoot = Join-Path $RepoRoot '.venv-refworld'
    $VenvPython = Join-Path $VenvRoot 'Scripts\python.exe'

    if (-not (Test-Path $VenvPython)) {
        Write-Host 'Creating repo-local Python environment (.venv-refworld)...'
        & $base.Command @($base.Prefix) -m venv $VenvRoot
        if ($LASTEXITCODE -ne 0) { throw 'Failed to create .venv-refworld.' }
    }

    Write-Host 'Preparing pinned CUDA Python environment...'
    & $VenvPython -m pip install --disable-pip-version-check --upgrade pip setuptools wheel
    if ($LASTEXITCODE -ne 0) { throw 'pip bootstrap failed.' }

    $cudaReady = $false
    & $VenvPython -c "import torch; assert torch.__version__.startswith('2.3.1'); assert torch.cuda.is_available()" *> $null
    if ($LASTEXITCODE -eq 0) { $cudaReady = $true }

    if (-not $cudaReady) {
        Write-Host 'Installing PyTorch 2.3.1 CUDA 12.1 wheels (first run is a large download)...'
        & $VenvPython -m pip install --upgrade --force-reinstall `
            torch==2.3.1 torchvision==0.18.1 `
            --index-url https://download.pytorch.org/whl/cu121
        if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed.' }
    }

    & $VenvPython -m pip install -e '.[dev,method]' huggingface_hub einops safetensors
    if ($LASTEXITCODE -ne 0) { throw 'RefWorld/VGGT dependency installation failed.' }

    & $VenvPython -c "import torch; print('torch', torch.__version__); print('cuda runtime', torch.version.cuda); print('gpu', torch.cuda.get_device_name(0)); assert torch.cuda.is_available()"
    if ($LASTEXITCODE -ne 0) {
        throw 'PyTorch cannot see CUDA. Update the NVIDIA driver and verify nvidia-smi, then rerun.'
    }

    $UpstreamRoot = Join-Path $RepoRoot '.upstream'
    $VggtRoot = Join-Path $UpstreamRoot 'vggt'
    New-Item -ItemType Directory -Force -Path $UpstreamRoot | Out-Null

    if (-not (Test-Path (Join-Path $VggtRoot '.git'))) {
        Write-Host 'Cloning pinned VGGT source...'
        & git clone $VggtRepo $VggtRoot
        if ($LASTEXITCODE -ne 0) { throw 'VGGT clone failed.' }
    }

    & git -C $VggtRoot cat-file -e "$VggtPin^{commit}" 2>$null
    if ($LASTEXITCODE -ne 0) {
        & git -C $VggtRoot fetch origin $VggtPin
        if ($LASTEXITCODE -ne 0) { throw 'Could not fetch the pinned VGGT commit.' }
    }
    & git -C $VggtRoot checkout --detach $VggtPin
    if ($LASTEXITCODE -ne 0) { throw 'Could not checkout the pinned VGGT commit.' }
    $head = (& git -C $VggtRoot rev-parse HEAD).Trim()
    if ($head -ne $VggtPin) { throw "VGGT pin verification failed: $head" }

    return [pscustomobject]@{ Python=$VenvPython; VggtRoot=$VggtRoot }
}

function Invoke-NativeSmoke([int]$Size, $Native) {
    $VenvPython = $Native.Python
    $VggtRoot = $Native.VggtRoot
    $SourceDir = Join-Path $OutputHost 'source-geometry'
    $SplatDir = Join-Path $OutputHost 'source-splat'
    $WarpDir = Join-Path $OutputHost 'warp-only'
    New-Item -ItemType Directory -Force -Path $SourceDir,$SplatDir,$WarpDir | Out-Null

    $LogPath = Join-Path $OutputHost "gpu-run-$Size.log"
    $allLog = New-Object System.Collections.Generic.List[string]

    function Run-Step([string]$Name, [string[]]$Args) {
        Write-Host $Name -ForegroundColor Cyan
        $lines = & $VenvPython @Args 2>&1 | Tee-Object -FilePath $LogPath -Append
        foreach ($line in $lines) { [void]$allLog.Add([string]$line) }
        return $LASTEXITCODE
    }

    $env:PYTORCH_CUDA_ALLOC_CONF = 'max_split_size_mb:128'

    if (-not $SkipTests) {
        $code = Run-Step 'Running focused contract tests...' @(
            '-m','pytest','-q',
            'tests/test_vggt_source.py','tests/test_source_geometry.py','tests/test_pinhole_warp.py',
            'tests/test_proposals.py','tests/test_splats.py','tests/test_schemas.py'
        )
        if ($code -ne 0) { return [pscustomobject]@{ Code=$code; Log=($allLog -join "`n") } }
    }

    $code = Run-Step "Running REAL VGGT inference at ${Size}x${Size}..." @(
        '-m','refworld.runners.vggt_source',
        '--vggt-root',$VggtRoot,
        '--reference',$ReferencePath,
        '--output',$SourceDir,
        '--seed','0',
        '--model-size',[string]$Size
    )
    if ($code -ne 0) { return [pscustomobject]@{ Code=$code; Log=($allLog -join "`n") } }

    $code = Run-Step 'Building source-only 3DGS diagnostic...' @(
        '-m','refworld.runners.source_splat',
        '--reference',$ReferencePath,
        '--source-geometry',(Join-Path $SourceDir 'source-geometry.safe.json'),
        '--output',$SplatDir
    )
    if ($code -ne 0) { return [pscustomobject]@{ Code=$code; Log=($allLog -join "`n") } }

    $code = Run-Step 'Generating warp-only near-view neighborhood...' @(
        '-m','refworld.runners.warp_only',
        '--reference',$ReferencePath,
        '--source-geometry',(Join-Path $SourceDir 'source-geometry.safe.json'),
        '--output',$WarpDir
    )
    return [pscustomobject]@{ Code=$code; Log=($allLog -join "`n") }
}

$dockerAvailable = (-not $ForceNative) -and [bool](Get-Command 'docker' -ErrorAction SilentlyContinue)
$Backend = if ($dockerAvailable) { 'Docker CUDA' } else { 'native Windows CUDA' }
$Native = $null
if (-not $dockerAvailable) {
    $Native = Invoke-NativeSetup
}

function Invoke-SelectedSmoke([int]$Size) {
    if ($dockerAvailable) { return Invoke-DockerSmoke $Size }
    return Invoke-NativeSmoke $Size $Native
}

$result = Invoke-SelectedSmoke $ChosenSize
if ($result.Code -ne 0) {
    $looksLikeOom = $result.Log -match '(?i)(cuda.*out of memory|out of memory.*cuda|CUDNN_STATUS_ALLOC_FAILED)'
    if ($looksLikeOom -and $ChosenSize -gt 336) {
        $FallbackSize = 336
        Write-Warning "CUDA OOM at $ChosenSize. Retrying once at smoke-only ${FallbackSize}x${FallbackSize}."
        Remove-Item (Join-Path $OutputHost 'source-geometry') -Recurse -Force -ErrorAction SilentlyContinue
        Remove-Item (Join-Path $OutputHost 'source-splat') -Recurse -Force -ErrorAction SilentlyContinue
        Remove-Item (Join-Path $OutputHost 'warp-only') -Recurse -Force -ErrorAction SilentlyContinue
        $result = Invoke-SelectedSmoke $FallbackSize
        $ChosenSize = $FallbackSize
    }
}
if ($result.Code -ne 0) {
    throw "Inference failed. See $OutputHost\gpu-run-$ChosenSize.log"
}

$ManifestPath = Join-Path $OutputHost 'source-geometry\source-geometry.safe.json'
Show-Result $ManifestPath $ChosenSize $Backend
