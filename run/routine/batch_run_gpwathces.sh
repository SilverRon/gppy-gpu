#!/bin/bash

# 반복 횟수 설정
NUM_REPETITIONS=15

# 첫 번째 탭부터 반복하여 실행
for i in $(seq 1 $NUM_REPETITIONS); do
    gnome-terminal --tab -- bash -c "python gpwatch_7DT_gain2750_test.py 7DT$(printf "%02d" $i); exec bash"
    sleep 1  # 각 탭 실행 사이에 N초 대기
done

