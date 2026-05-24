#!/bin/bash
# monitor_traces.sh — Monitor SDM trace collection progress.
#
# Usage:
#   bash ~/Ditto/src/trace_collection/monitor_traces.sh
#
# Shows: process status, log tail, disk usage, current image count.

TRACE_ROOT="$HOME/Ditto/traces/sdm"
LOG_FILE="$TRACE_ROOT/collection.log"
PID_FILE="$TRACE_ROOT/collection.pid"

clear
echo "=== SDM Trace Collection Monitor ==="
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 1. Process status
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        ELAPSED=$(ps -o etime= -p "$PID" | xargs)
        CPU=$(ps -o %cpu= -p "$PID" | xargs)
        MEM_GB=$(ps -o rss= -p "$PID" | awk '{printf "%.1f", $1/1024/1024}')
        echo "Process: ✓ RUNNING (PID $PID, $ELAPSED elapsed, ${CPU}% CPU, ${MEM_GB} GB RAM)"
    else
        echo "Process: ✗ NOT RUNNING (PID $PID stale; collection finished or crashed)"
    fi
else
    echo "Process: ✗ PID file missing — not launched yet?"
fi
echo ""

# 2. Image count
if [ -d "$TRACE_ROOT" ]; then
    IMG_COUNT=$(find "$TRACE_ROOT" -name "DONE" 2>/dev/null | wc -l)
    DIR_COUNT=$(find "$TRACE_ROOT" -maxdepth 1 -type d -name "image_*" 2>/dev/null | wc -l)
    echo "Images:  $IMG_COUNT / 20 complete  ($DIR_COUNT directories on disk)"
else
    echo "Images:  0 / 20 (output dir does not exist yet)"
fi

# 3. Disk usage
if [ -d "$TRACE_ROOT" ]; then
    SIZE=$(du -sh "$TRACE_ROOT" 2>/dev/null | cut -f1)
    FILES=$(find "$TRACE_ROOT" -name "*.npz" 2>/dev/null | wc -l)
    echo "Storage: $SIZE in $FILES .npz files"
fi
echo ""

# 4. GPU status
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "=== GPU ==="
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw \
               --format=csv,noheader,nounits | \
    awk -F', ' '{printf "  %-30s  %s%% GPU   %s/%s MB   %sW\n", $1, $2, $3, $4, $5}'
    echo ""
fi

# 5. Recent log
if [ -f "$LOG_FILE" ]; then
    echo "=== Recent log (last 15 lines) ==="
    tail -15 "$LOG_FILE"
else
    echo "Log file not found: $LOG_FILE"
fi
