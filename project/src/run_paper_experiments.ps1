param(
    [int]$WaitPid = 0,
    [switch]$ContinueOnError,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$SrcDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$WorkspaceRoot = Resolve-Path (Join-Path $SrcDir "..\..")
$ArtifactsDir = Join-Path $WorkspaceRoot "artifacts"
$LogDir = Join-Path $ArtifactsDir "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$QueueLog = Join-Path $LogDir "paper_experiment_queue.log"
$QueueCsv = Join-Path $LogDir "paper_experiment_queue.csv"

function Write-QueueLog {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$timestamp] $Message"
    Write-Host $line
    Add-Content -Path $QueueLog -Value $line
}

function Write-QueueCsv {
    param(
        [string]$Name,
        [string]$Status,
        [int]$ExitCode,
        [string]$OutLog,
        [string]$ErrLog
    )
    if (-not (Test-Path $QueueCsv)) {
        Add-Content -Path $QueueCsv -Value "timestamp,name,status,exit_code,out_log,err_log"
    }
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = '"{0}","{1}","{2}",{3},"{4}","{5}"' -f $timestamp, $Name, $Status, $ExitCode, $OutLog, $ErrLog
    Add-Content -Path $QueueCsv -Value $line
}

function ResultPath {
    param([string]$Dataset, [string]$Augment, [string]$Model)
    $suffix = if ($Augment -eq "none") { "" } else { "_$Augment" }
    return Join-Path $ArtifactsDir "results_${Dataset}${suffix}\${Model}_flow_kfold_results.json"
}

function EvalPath {
    param([string]$Dataset, [string]$Augment)
    $suffix = if ($Augment -eq "none") { "" } else { "_$Augment" }
    return Join-Path $ArtifactsDir "results_${Dataset}${suffix}\eval_results.json"
}

function ModelPath {
    param([string]$Dataset, [string]$Augment, [string]$ModelDir, [string]$File)
    $suffix = if ($Augment -eq "none") { "" } else { "_$Augment" }
    return Join-Path $ArtifactsDir "models_${Dataset}${suffix}\${ModelDir}\${File}"
}

function GeneratorPath {
    param([string]$Dataset, [string]$Augment)
    if ($Augment -eq "gan") {
        return Join-Path $WorkspaceRoot "project\data\processed\${Dataset}_gan\generator.pt"
    }
    return Join-Path $WorkspaceRoot "project\data\processed\${Dataset}_wcgan_gp\generator_wcgan_gp.pt"
}

function Test-Outputs {
    param([string[]]$Outputs)
    foreach ($output in $Outputs) {
        if (-not (Test-Path $output)) {
            return $false
        }
        $item = Get-Item $output
        if ($item.Length -le 0) {
            return $false
        }
    }
    return $true
}

function Test-MinF1 {
    param([string]$JsonPath, [double]$MinF1)
    if (-not (Test-Path $JsonPath)) {
        return $false
    }
    try {
        $data = Get-Content -Raw -Path $JsonPath | ConvertFrom-Json
        $f1 = [double]$data.summary.f1.mean
        return ($f1 -ge $MinF1)
    } catch {
        return $false
    }
}

function Invoke-QueuedPython {
    param([hashtable]$Job)

    $name = [string]$Job["Name"]
    $args = [string[]]$Job["Args"]
    $outputs = @()
    if ($Job.ContainsKey("Outputs")) {
        $outputs = [string[]]$Job["Outputs"]
    }
    $force = $false
    if ($Job.ContainsKey("Force")) {
        $force = [bool]$Job["Force"]
    }
    $minF1 = $null
    if ($Job.ContainsKey("MinF1")) {
        $minF1 = [double]$Job["MinF1"]
    }

    if (-not $force -and $outputs.Count -gt 0 -and (Test-Outputs -Outputs $outputs)) {
        if ($null -eq $minF1 -or (Test-MinF1 -JsonPath $outputs[0] -MinF1 $minF1)) {
            Write-QueueLog "SKIP $name (outputs already present)"
            Write-QueueCsv -Name $name -Status "skipped" -ExitCode 0 -OutLog "" -ErrLog ""
            return
        }
        Write-QueueLog "RERUN $name (existing result below min F1 $minF1)"
    }

    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $safeName = $name -replace "[^A-Za-z0-9_.-]", "_"
    $outLog = Join-Path $LogDir "${safeName}_${stamp}.out.log"
    $errLog = Join-Path $LogDir "${safeName}_${stamp}.err.log"

    Write-QueueLog "START $name :: python $($args -join ' ')"
    $process = Start-Process `
        -FilePath "python" `
        -ArgumentList (@("-u") + $args) `
        -WorkingDirectory $SrcDir `
        -RedirectStandardOutput $outLog `
        -RedirectStandardError $errLog `
        -WindowStyle Hidden `
        -Wait `
        -PassThru

    if ($process.ExitCode -ne 0) {
        Write-QueueLog "FAIL $name exit=$($process.ExitCode); stderr=$errLog"
        Write-QueueCsv -Name $name -Status "failed" -ExitCode $process.ExitCode -OutLog $outLog -ErrLog $errLog
        if (-not $ContinueOnError) {
            throw "$name failed with exit code $($process.ExitCode)"
        }
        return
    }

    Write-QueueLog "DONE $name"
    Write-QueueCsv -Name $name -Status "done" -ExitCode 0 -OutLog $outLog -ErrLog $errLog
}

function Add-TrainJob {
    param(
        [System.Collections.ArrayList]$Jobs,
        [string]$Dataset,
        [string]$Augment,
        [string]$Script,
        [string]$Model,
        [string]$ModelDir,
        [string]$ModelFile,
        [switch]$TargetZero,
        [double]$MinF1 = -1.0
    )
    $args = @($Script, "--dataset", $Dataset)
    if ($Augment -ne "none") {
        $args += @("--augment", $Augment)
    }
    if ($TargetZero) {
        $args += @("--target_bot_ratio", "0")
    }
    $job = @{
        Name = "${Dataset}_${Augment}_${Model}"
        Args = $args
        Outputs = @(
            (ResultPath -Dataset $Dataset -Augment $Augment -Model $Model),
            (ModelPath -Dataset $Dataset -Augment $Augment -ModelDir $ModelDir -File $ModelFile)
        )
    }
    if ($MinF1 -ge 0) {
        $job["MinF1"] = $MinF1
    }
    [void]$Jobs.Add($job)
}

function Add-EvaluateJob {
    param([System.Collections.ArrayList]$Jobs, [string]$Dataset, [string]$Augment)
    $args = @("evaluate.py", "--dataset", $Dataset)
    if ($Augment -ne "none") {
        $args += @("--augment", $Augment)
    }
    [void]$Jobs.Add(@{
        Name = "${Dataset}_${Augment}_evaluate"
        Args = $args
        Outputs = @(EvalPath -Dataset $Dataset -Augment $Augment)
        Force = $true
    })
}

function Add-GeneratorJob {
    param([System.Collections.ArrayList]$Jobs, [string]$Dataset, [string]$Augment)
    $script = if ($Augment -eq "gan") { "augment_gan.py" } else { "augment_wcgan_gp.py" }
    [void]$Jobs.Add(@{
        Name = "${Dataset}_${Augment}_generator"
        Args = @($script, "--dataset", $Dataset)
        Outputs = @(GeneratorPath -Dataset $Dataset -Augment $Augment)
    })
}

$Jobs = [System.Collections.ArrayList]::new()
$Datasets = @("cicids2017", "cicids2018", "ctu13")
$Augments = @("none", "smote", "gan", "wcgan_gp")
$TrainSpecs = @(
    @{ Script = "train_rf.py";       Model = "rf";       ModelDir = "rf";       ModelFile = "rf_flow.pkl" },
    @{ Script = "train_xgb.py";      Model = "xgb";      ModelDir = "xgb";      ModelFile = "xgb_flow.pkl" },
    @{ Script = "train_cnn_lstm.py"; Model = "cnn_lstm"; ModelDir = "cnn_lstm"; ModelFile = "cnn_lstm_flow.pt" },
    @{ Script = "train_gru.py";      Model = "gru";      ModelDir = "gru";      ModelFile = "gru_flow.pt" },
    @{ Script = "train_cnn_gru.py";  Model = "cnn_gru";  ModelDir = "cnn_gru";  ModelFile = "cnn_gru_flow.pt" }
)

foreach ($augment in $Augments) {
    if ($augment -in @("gan", "wcgan_gp")) {
        foreach ($dataset in $Datasets) {
            Add-GeneratorJob -Jobs $Jobs -Dataset $dataset -Augment $augment
        }
    }

    foreach ($dataset in $Datasets) {
        foreach ($spec in $TrainSpecs) {
            $isDeep = $spec.Model -in @("cnn_lstm", "gru", "cnn_gru")
            $targetZero = ($dataset -eq "cicids2017" -and $isDeep)
            $minF1 = if ($augment -eq "none" -and $dataset -eq "cicids2017" -and $isDeep) { 0.2 } else { -1.0 }
            Add-TrainJob `
                -Jobs $Jobs `
                -Dataset $dataset `
                -Augment $augment `
                -Script $spec.Script `
                -Model $spec.Model `
                -ModelDir $spec.ModelDir `
                -ModelFile $spec.ModelFile `
                -TargetZero:($targetZero) `
                -MinF1 $minF1
        }
        Add-EvaluateJob -Jobs $Jobs -Dataset $dataset -Augment $augment
    }
}

foreach ($augment in $Augments) {
    $args = @("visualize.py")
    if ($augment -ne "none") {
        $args += @("--augment", $augment)
    }
    [void]$Jobs.Add(@{
        Name = "visualize_${augment}"
        Args = $args
    })
}

[void]$Jobs.Add(@{
    Name = "summarize_experiments"
    Args = @("summarize_experiments.py")
    Outputs = @(
        (Join-Path $ArtifactsDir "paper_summary.md"),
        (Join-Path $ArtifactsDir "paper_summary.csv")
    )
    Force = $true
})

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

Write-QueueLog "Paper experiment queue started. Jobs=$($Jobs.Count)"
if ($DryRun) {
    foreach ($job in $Jobs) {
        Write-QueueLog "DRYRUN $($job["Name"]) :: python $([string[]]$job["Args"] -join ' ')"
    }
    Write-QueueLog "Dry run finished"
    return
}
foreach ($job in $Jobs) {
    Invoke-QueuedPython -Job $job
}
Write-QueueLog "Paper experiment queue finished"
