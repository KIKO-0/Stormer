#!/bin/bash
# watchdog.sh: 监控下载进程，卡死自动重启

DOWNLOAD_SCRIPT="/home/zhangbo/stormer/download_full_data.py"
LOG_FILE="/home/stormer_data/download.log"
PYTHON="${PYTHON:-/root/miniconda3/envs/tnp/bin/python}"
CHECK_INTERVAL=300  # 每5分钟检查一次
STALE_THRESHOLD=12  # 连续12次无变化则判定卡死（60分钟）

get_pid() {
    # 只取一个 PID，避免多行导致 ps/kill 异常
    pgrep -f "download_full_data.py" 2>/dev/null | head -n1
}

get_mem() {
    local pid=$1
    ps -o rss= -p $pid 2>/dev/null
}

start_download() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 启动下载进程..." >> $LOG_FILE
    nohup "$PYTHON" -u $DOWNLOAD_SCRIPT >> $LOG_FILE 2>&1 &
    echo $!
}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Watchdog 启动"

prev_mem=0
stale_count=0

while true; do
    pid=$(get_pid)

    if [ -z "$pid" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 进程不存在，重启..."
        start_download
        prev_mem=0
        stale_count=0
        sleep $CHECK_INTERVAL
        continue
    fi

    curr_mem=$(get_mem $pid)

    if [ "$curr_mem" = "$prev_mem" ]; then
        stale_count=$((stale_count + 1))
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 内存无变化 ($curr_mem KB)，连续 $stale_count 次"
        if [ $stale_count -ge $STALE_THRESHOLD ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] 判定卡死，kill $pid 并重启..."
            kill $pid
            sleep 5
            start_download
            prev_mem=0
            stale_count=0
        fi
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 正常运行，内存 $prev_mem -> $curr_mem KB"
        stale_count=0
        prev_mem=$curr_mem
    fi

    sleep $CHECK_INTERVAL
done
