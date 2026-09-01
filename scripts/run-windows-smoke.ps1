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

function Require-Command([string]$Name, [string]$Hint) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "$Name is required. $Hint"
    }
}

Require-Command 'docker' 'Install/start Docker Desktop with the WSL2 backend.'
Require-Command 'nvidia-smi' 'Install/update the NVIDIA Windows driver.'

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

Write-Host 'Checking Docker GPU passthrough...'
& docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
if ($LASTEXITCODE -ne 0) {
    throw 'Docker cannot see the NVIDIA GPU. Ensure Docker Desktop is using WSL2 and GPU support is enabled.'
}

$OutputRelative = "outputs/windows-smoke/$RunName"
$OutputHost = Join-Path $RepoRoot ($OutputRelative -replace '/', [IO.Path]::DirectorySeparatorChar)
New-Item -ItemType Directory -Force -Path $OutputHost | Out-Null

$HfCache = Join-Path $env:USERPROFILE '.cache\huggingface'
New-Item -ItemType Directory -Force -Path $HfCache | Out-Null

Write-Host 'Building pinned VGGT container...'
& docker build -f docker/vggt.Dockerfile -t refworld-vggt .
if ($LASTEXITCODE -ne 0) { throw 'Docker image build failed.' }

$RunTests = if ($SkipTests) { '0' } else { '1' }

function Invoke-Smoke([int]$Size) {
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

    Write-Host "Running real VGGT inference at ${Size}x${Size}..." -ForegroundColor Cyan
    $log = & docker @dockerArgs 2>&1 | Tee-Object -FilePath (Join-Path $OutputHost "gpu-run-$Size.log")
    $code = $LASTEXITCODE
    return [pscustomobject]@{ Code=$code; Log=($log -join "`n") }
}

$result = Invoke-Smoke $ChosenSize
if ($result.Code -ne 0) {
    $looksLikeOom = $result.Log -match '(?i)(cuda.*out of memory|out of memory.*cuda|CUDNN_STATUS_ALLOC_FAILED)'
    if ($looksLikeOom -and $ChosenSize -gt 336) {
        $FallbackSize = 336
        Write-Warning "CUDA OOM at $ChosenSize. Retrying once at smoke-only ${FallbackSize}x${FallbackSize}."
        $result = Invoke-Smoke $FallbackSize
        $ChosenSize = $FallbackSize
    }
}
if ($result.Code -ne 0) {
    throw "Inference failed. See $OutputHost\gpu-run-$ChosenSize.log"
}

$ManifestPath = Join-Path $OutputHost 'source-geometry\source-geometry.safe.json'
if (-not (Test-Path $ManifestPath)) { throw 'Inference returned success but source-geometry manifest is missing.' }
$Manifest = Get-Content $ManifestPath -Raw | ConvertFrom-Json

$PeakGiB = [math]::Round(([double]$Manifest.environment.peak_reserved_bytes / 1GB), 2)
$InferSec = [math]::Round([double]$Manifest.timing.inference_seconds, 2)
$LoadSec = [math]::Round([double]$Manifest.timing.model_load_seconds, 2)

Write-Host ''
Write-Host 'REAL INFERENCE COMPLETE' -ForegroundColor Green
Write-Host "GPU:          $($Manifest.environment.gpu_name)"
Write-Host "Model size:   $($Manifest.preprocessing.model_size)"
Write-Host "Model load:   $LoadSec s"
Write-Host "Inference:    $InferSec s"
Write-Host "Peak reserved:$PeakGiB GiB"
Write-Host "Output:       $OutputHost"
Write-Host "Manifest:     $ManifestPath"
Write-Host ''
Write-Host 'Artifacts now include real VGGT depth/camera/confidence, a source-only 3DGS PLY, and warp-only near views.'
Write-Host 'Next: render source-splat/source-splat.ply at source-splat/source-camera.json and score it against the reference.'
