param(
    [int]$WaitPid = 0
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$SrcDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$WorkspaceRoot = Resolve-Path (Join-Path $SrcDir "..\..")
$LogDir = Join-Path $WorkspaceRoot "artifacts\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Write-QueueLog {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$timestamp] $Message"
    Write-Host $line
    Add-Content -Path (Join-Path $LogDir "remaining_baseline_queue.log") -Value $line
}

function Invoke-QueuedPython {
    param(
        [string]$Name,
        [string[]]$Args
    )

    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $safeName = $Name -replace "[^A-Za-z0-9_.-]", "_"
    $outLog = Join-Path $LogDir "${safeName}_${stamp}.out.log"
    $errLog = Join-Path $LogDir "${safeName}_${stamp}.err.log"

    Write-QueueLog "START $Name -> $outLog"
    $process = Start-Process `
        -FilePath "python" `
        -ArgumentList (@("-u") + $Args) `
        -WorkingDirectory $SrcDir `
        -RedirectStandardOutput $outLog `
        -RedirectStandardError $errLog `
        -WindowStyle Hidden `
        -Wait `
        -PassThru

    if ($process.ExitCode -ne 0) {
        Write-QueueLog "FAIL $Name exit=$($process.ExitCode); stderr=$errLog"
        throw "$Name failed with exit code $($process.ExitCode)"
    }

    Write-QueueLog "DONE $Name"
}

if ($WaitPid -gt 0) {
    $existing = Get-Process -Id $WaitPid -ErrorAction SilentlyContinue
    if ($null -ne $existing) {
        Write-QueueLog "Waiting for existing process PID=$WaitPid"
        Wait-Process -Id $WaitPid
        Write-QueueLog "Existing process PID=$WaitPid finished"
    } else {
        Write-QueueLog "Process PID=$WaitPid is not running; continuing"
    }
}

$jobs = @(
    @{ Name = "cicids2017_gru_target0"; Args = @("train_gru.py", "--dataset", "cicids2017", "--target_bot_ratio", "0") },
    @{ Name = "cicids2017_cnn_gru_target0"; Args = @("train_cnn_gru.py", "--dataset", "cicids2017", "--target_bot_ratio", "0") },
    @{ Name = "cicids2018_gru"; Args = @("train_gru.py", "--dataset", "cicids2018") },
    @{ Name = "cicids2018_cnn_gru"; Args = @("train_cnn_gru.py", "--dataset", "cicids2018") },
    @{ Name = "ctu13_rf"; Args = @("train_rf.py", "--dataset", "ctu13") },
    @{ Name = "ctu13_xgb"; Args = @("train_xgb.py", "--dataset", "ctu13") },
    @{ Name = "ctu13_cnn_lstm"; Args = @("train_cnn_lstm.py", "--dataset", "ctu13") },
    @{ Name = "ctu13_gru"; Args = @("train_gru.py", "--dataset", "ctu13") },
    @{ Name = "ctu13_cnn_gru"; Args = @("train_cnn_gru.py", "--dataset", "ctu13") },
    @{ Name = "evaluate_cicids2017"; Args = @("evaluate.py", "--dataset", "cicids2017") },
    @{ Name = "evaluate_cicids2018"; Args = @("evaluate.py", "--dataset", "cicids2018") },
    @{ Name = "evaluate_ctu13"; Args = @("evaluate.py", "--dataset", "ctu13") }
)

foreach ($job in $jobs) {
    Invoke-QueuedPython -Name $job["Name"] -Args $job["Args"]
}

Write-QueueLog "All remaining baseline jobs finished"
