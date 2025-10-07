param(
  [string]$Root = (Resolve-Path "."),
  [ValidateSet("report","quarantine","delete")]
  [string]$Action = "report",
  [int]$LargeMB = 50,
  [string]$Quarantine = ".\artifacts\housekeeping\quarantine"
)

function Join-Norm([string]$a, [string]$b) { [IO.Path]::GetFullPath((Join-Path $a $b)) }

$Root = (Resolve-Path $Root).Path
$OutDir = Join-Norm $Root ".\artifacts\housekeeping"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$ReportCsv = Join-Path $OutDir "thrift_report.csv"
$SummaryTxt = Join-Path $OutDir "summary.txt"

# locate git (optional)
$Git = $null
foreach ($p in @("C:\Program Files\Git\cmd\git.exe","C:\Program Files (x86)\Git\cmd\git.exe")) {
  if (Test-Path $p) { $Git = $p; break }
}
$HasGit = $Git -ne $null -and (Test-Path (Join-Path $Root ".git"))

# classification globs (tune if needed)
$KeepGlobs = @(
  "familook\service\search_service.py",
  "familook\service\labels.csv",
  "familook\src\config\*.yaml",
  "familook\src\**\*.py",
  "familook\**\*.py",
  "scripts\**\*.py",
  "curated\_filter_strict_good.py",
  "requirements*.txt",
  "pyproject.toml",
  "README.md","LICENSE",".gitignore"
)
$GeneratedLargeGlobs = @(
  "artifacts\**\*",
  "curated\good_frontal_*",
  "curated\final_*",
  "curated\*\unique*",
  "curated\removed_outliers_*",
  ".venv\**\*",
  ".insightface\**\*",
  "**\*.npy","**\*.npz","**\*.pt","**\*.onnx",
  "**\*.zip","**\*.7z","**\*.rar"
)
$DevOnlyGlobs = @(".vscode\**\*",".idea\**\*","**\*.log","**\*.tmp","**\*.bak")

function Test-Glob([string]$Path,[string[]]$Globs){
  $rel=(Resolve-Path $Path -Relative).TrimStart(".\")
  foreach($g in $Globs){
    $rx="^"+([Regex]::Escape($g) -replace "\\\*\\\*","(.+)" -replace "\\\*","([^\\\/]+)")+"$"
    if($rel -match $rx){ return $true }
  }
  return $false
}

$files = Get-ChildItem -Path $Root -Recurse -File -Force | Where-Object { $_.FullName -notmatch "\\\.git\\" }
$rows = New-Object System.Collections.Generic.List[object]
$largeThresh = $LargeMB * 1MB
$counts = @{ keep=0; gen=0; dev=0; unknown=0; size_keep=0L; size_gen=0L; size_dev=0L; size_unknown=0L }

foreach($f in $files){
  $path=$f.FullName
  $rel=$path.Substring($Root.Length).TrimStart("\","/")
  $size=$f.Length
  $age=(Get-Date)-$f.LastWriteTime
  $cat="unknown"
  if(Test-Glob $path $KeepGlobs){ $cat="keep" }
  elseif(Test-Glob $path $GeneratedLargeGlobs){ $cat="gen" }
  elseif(Test-Glob $path $DevOnlyGlobs){ $cat="dev" }
  $isLarge = $size -ge $largeThresh

  $gitIgnored=$false
  if($HasGit){
    $relGit=$rel -replace "\\","/"
    $psi=New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName=$Git; $psi.WorkingDirectory=$Root
    $psi.Arguments="check-ignore --quiet -- '$relGit'"
    $psi.UseShellExecute=$false; $psi.RedirectStandardError=$true; $psi.RedirectStandardOutput=$true
    $proc=[System.Diagnostics.Process]::Start($psi); $proc.WaitForExit()
    $gitIgnored=($proc.ExitCode -eq 0)
  }

  $rows.Add([pscustomobject]@{
    Category=$cat; GitIgnored=$gitIgnored; LargeMB=[Math]::Round($size/1MB,2); IsLarge=$isLarge
    LastWrite=$f.LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss"); AgeDays=[Math]::Round($age.TotalDays,1)
    RelPath=$rel; FullPath=$path
  })

  switch($cat){
    "keep"  { $counts.keep++;  $counts.size_keep  += $size }
    "gen"   { $counts.gen++;   $counts.size_gen   += $size }
    "dev"   { $counts.dev++;   $counts.size_dev   += $size }
    default { $counts.unknown++; $counts.size_unknown += $size }
  }
}

$rows | Sort-Object Category,RelPath | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $ReportCsv
$summary = @(
  "Files: keep=$($counts.keep), gen=$($counts.gen), dev=$($counts.dev), unknown=$($counts.unknown)",
  ("Sizes (MB): keep={0:N1}, gen={1:N1}, dev={2:N1}, unknown={3:N1}" -f `
    ($counts.size_keep/1MB), ($counts.size_gen/1MB), ($counts.size_dev/1MB), ($counts.size_unknown/1MB)),
  "Report: $ReportCsv"
)
$summaryText = $summary -join [Environment]::NewLine
$summaryText | Set-Content -Encoding UTF8 $SummaryTxt
Write-Host $summaryText

if($Action -in @("quarantine","delete")){
  $targets = $rows | Where-Object { $_.Category -in @("gen","dev") -or ($_.Category -eq "unknown" -and $_.IsLarge) }
  if($Action -eq "quarantine"){
    $qDir = Join-Norm $Root $Quarantine
    Write-Host "Quarantining $($targets.Count) items to $qDir ..."
    foreach($t in $targets){
      $dest = Join-Norm $qDir $t.RelPath
      New-Item -ItemType Directory -Force -Path (Split-Path $dest) | Out-Null
      Move-Item -Force -Path $t.FullPath -Destination $dest
    }
  } else {
    Write-Host "Deleting $($targets.Count) items ..."
    foreach($t in $targets){ Remove-Item -Force -Path $t.FullPath }
  }
  Write-Host "Done."
}

Write-Host "`nTop 20 largest files:"
$rows | Sort-Object -Property LargeMB -Descending | Select-Object -First 20 | Format-Table LargeMB,Category,GitIgnored,RelPath
