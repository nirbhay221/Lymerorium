# Jetson llama-server watchdog (external probe only).
# Polls JETSON_IP:8080 every $IntervalSec, appends status to $Log.
# Detects HTTP 500 / OOM / refused / timeout and latency creep.
# Log columns: ISO_time, STATUS, latency_ms, detail

$Jetson      = if ($env:JETSON_IP) { $env:JETSON_IP } else { Read-Host "Enter Jetson IP" }
$Url         = "http://$Jetson`:8080/v1/chat/completions"
$Model       = "unsloth/gemma-4-E2B-it-GGUF:Q4_K_M"
$Log         = "C:\Users\nirbh\projects\vision-app\jetson_watchdog.log"
$IntervalSec = 30
$SlowMs      = 15000     # an 8-token gen this slow = RAM climbing, early warning
$TimeoutSec  = 45

$body = @{ model=$Model; messages=@(@{role="user";content="ping"}); max_tokens=4 } | ConvertTo-Json -Compress
$last = ""

function Write-Line($status, $ms, $detail) {
  $ts = (Get-Date).ToString("o")
  $line = "{0},{1},{2},{3}" -f $ts, $status, $ms, $detail
  Add-Content -Path $Log -Value $line -Encoding utf8
}

Write-Line "START" 0 "watchdog up, interval=${IntervalSec}s slow=${SlowMs}ms"

while ($true) {
  $alive = Test-Connection -ComputerName $Jetson -Count 1 -Quiet -ErrorAction SilentlyContinue
  if (-not $alive) {
    Write-Line "DOWN" 0 "no ping reply (host unreachable)"
    if ($last -ne "DOWN") { Write-Line "BLOWUP" 0 "Jetson stopped responding to ping" }
    $last = "DOWN"
    Start-Sleep -Seconds $IntervalSec; continue
  }

  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  try {
    $r = Invoke-RestMethod -Uri $Url -Method Post -Body $body -ContentType "application/json" -TimeoutSec $TimeoutSec
    $sw.Stop(); $ms = [int]$sw.ElapsedMilliseconds
    if ($ms -ge $SlowMs) {
      Write-Line "SLOW" $ms "gen ok but latency high (RAM likely climbing)"
      if ($last -ne "SLOW" -and $last -ne "DOWN" -and $last -ne "ERR500") { Write-Line "WARN" $ms "latency creep started" }
      $last = "SLOW"
    } else {
      if ($last -ne "OK") { Write-Line "RECOVER" $ms "back to normal" }
      Write-Line "OK" $ms "gen ok"
      $last = "OK"
    }
  } catch {
    $sw.Stop(); $ms = [int]$sw.ElapsedMilliseconds
    $msg = $_.Exception.Message
    $code = ""
    if ($_.Exception.Response) { try { $code = [int]$_.Exception.Response.StatusCode } catch {} }
    if ($code -eq 500 -or $msg -match "500") {
      Write-Line "ERR500" $ms "HTTP 500 (OOM / runner terminated)"
      if ($last -ne "ERR500") { Write-Line "BLOWUP" $ms "HTTP 500 from llama-server - Gemma-4 OOM" }
      $last = "ERR500"
    } else {
      Write-Line "DOWN" $ms ("request failed: " + $msg.Replace("`n"," ").Replace("`r"," "))
      if ($last -ne "DOWN") { Write-Line "BLOWUP" $ms ("llama-server unreachable: " + $msg.Replace("`n"," ").Replace("`r"," ")) }
      $last = "DOWN"
    }
  }
  Start-Sleep -Seconds $IntervalSec
}
