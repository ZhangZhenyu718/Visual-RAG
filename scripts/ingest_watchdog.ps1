Set-Location "D:\MyCode\Visual-RAG"
$log = "artifacts\ingest_val.log"
$max_restarts = 30
$n = 0
while ($n -lt $max_restarts) {
    if (Select-String -Path $log -Pattern "\[ingest\] done" -Quiet) { break }
    Add-Content $log "[watchdog] (re)starting ingest, attempt $($n + 1)"
    cmd /c "D:\Anaconda\envs\visualrag\python.exe -u scripts\ingest_dataset.py --split val >> artifacts\ingest_val.log 2>&1"
    $n++
    Start-Sleep -Seconds 5
}
Add-Content $log "[watchdog] exiting after $n attempts"
