#!/usr/bin/env bash

# 파이썬 파이프라인 실행
python 7DT_Routine_1x1_gain2750.py 7DT02 /lyman/data1/obsdata/7DT02/2024-11-16_gain2750 &

# 백그라운드로 실행된 프로세스의 PID 획득
PIPELINE_PID=$!

# 메모리 사용량 로깅을 위한 로그파일 지정
LOGFILE="/lyman/data1/factory/test/track_memory/pipeline_memory_usage.log"

# 초기 로그 헤더
echo "Timestamp, RSS(KB), VSZ(KB), CMD" > $LOGFILE

# 프로세스가 종료될 때까지 일정 간격으로 메모리 사용량 기록
while kill -0 "$PIPELINE_PID" 2>/dev/null; do
    # ps를 이용한 프로세스 메모리 조회 (RSS, VSZ 확인)
    MEM_INFO=$(ps -p $PIPELINE_PID -o rss,vsz,cmd --no-headers)
    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
    echo "$TIMESTAMP, $MEM_INFO" >> $LOGFILE

    # 5초 대기 후 재측정
    sleep 5
done

echo "Pipeline process ($PIPELINE_PID) ended."
echo "Check $LOGFILE for memory usage history."
