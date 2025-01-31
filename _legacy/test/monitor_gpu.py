import subprocess
import pandas as pd
import time
from datetime import datetime

def get_gpu_usage():
    # nvidia-smi 명령어 실행 및 출력받기
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu',
                             '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8').strip().split('\n')
    
    # 데이터를 파싱하여 DataFrame으로 변환
    data = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    for line in output:
        gpu_info = line.split(', ')
        gpu_info.insert(0, timestamp)  # 타임스탬프 추가
        data.append(gpu_info)
    
    # 컬럼 이름: 타임스탬프, GPU index, 이름, 메모리 총량, 사용 중인 메모리, GPU 사용률, 온도
    columns = ['Timestamp', 'GPU Index', 'GPU Name', 'Memory Total (MB)', 'Memory Used (MB)', 'GPU Utilization (%)', 'Temperature (C)']
    df = pd.DataFrame(data, columns=columns)
    
    return df

def log_gpu_usage(log_file, interval=5):
    while True:
        df = get_gpu_usage()
        print(df)
        
        # 여기서 실시간으로 테이블을 업데이트하거나 로그 파일에 저장 가능
        # 예시로 DataFrame을 특정 파일에 저장하려면 아래와 같이 할 수 있음:
        df.to_csv(log_file, mode='a', header=False)
        
        time.sleep(interval)

#

interval = 0.5
log_file = '/large_data/factory/log/gpu.log'

# GPU 사용률을 5초마다 기록
log_gpu_usage(log_file=log_file, interval=interval)
