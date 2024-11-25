#============================================================
import os
import glob
import sys
import time
import json
import gc
import subprocess
from pathlib import Path
from astropy.table import Table
#============================================================
#   Function
#------------------------------------------------------------
def get_folder_size(folder_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total_size += os.path.getsize(fp)
    return total_size
#------------------------------------------------------------
def format_size(size_in_bytes):
    # KB, MB, GB 규정으로 변환
    size_in_kb = size_in_bytes / 1024
    size_in_mb = size_in_kb / 1024
    size_in_gb = size_in_mb / 1024

    if size_in_gb >= 1:
        return f"{size_in_gb:.2f} GB"
    elif size_in_mb >= 1:
        return f"{size_in_mb:.2f} MB"
    else:
        return f"{size_in_kb:.2f} KB"
#------------------------------------------------------------
def get_input():
    if len(sys.argv) > 1:
        return sys.argv[1]
    else:
        return input("OBS? (e.g. 7DT01): ")
#------------------------------------------------------------
obs = get_input()
print(f"="*60)
print(f"Start to Monitor {obs} Data")
# obs = '7DT01'
# print(f"OBS: {obs}")
print(f"-"*60)

interval_monitor = 30
#------------------------------------------------------------
#   Path
#------------------------------------------------------------
path_thisfile = Path(__file__).resolve()
path_root = path_thisfile.parent.parent.parent  # Careful! not a str
with open(path_thisfile.parent / 'path.json', 'r') as jsonfile:
    upaths = json.load(jsonfile)

# paths from path.json
path_base = upaths['path_base']  # '/large_data/factory'
path_obsdata = f'{path_base}/../obsdata' if upaths['path_obsdata'] == '' else upaths['path_obsdata']
# path_gal = f'{path_base}/../processed_{n_binning}x{n_binning}_gain2750' if upaths['path_gal'] == '' else upaths['path_gal']
path_refcat = f'{path_base}/ref_cat' if upaths['path_refcat'] == '' else upaths['path_refcat']
# path_to_obsdata = f"/large_data/obsdata"
path_to_monitor = str(Path(path_obsdata) / obs)  # f"{path_to_obsdata}/{obs}"
path_to_log = str(Path(path_base) / 'log')  # f"/large_data/factory/log"
# path_to_gppy = f"/home/gp/gppy"
path_run = path_root / 'run' / 'routine'
code = "7DT_Routine_1x1_gain2750.py"
logfile = f"{path_to_log}/{obs.lower()}.log"
#------------------------------------------------------------
while True:
    #------------------------------------------------------------
    #   Log Table
    #------------------------------------------------------------
    if os.path.exists(logfile):
        logtbl = Table.read(logfile, format='csv')
    else:
        print(f"No {logfile}. Generate default log table.")
        os.system(f"cp {path_to_log}/7dt.log {logfile}")
        logtbl = Table.read(logfile, format='csv')
    #------------------------------------------------------------
    #   DATE Folders (=Raw Data)
    #------------------------------------------------------------
    pattern_survey  = "20??-??-??_gain2750"
    pattern_too = "20??-??-??_ToO_gain2750"
    pattern_test = "20??-00-??"

    survey_data_list = sorted(glob.glob(f"{path_to_monitor}/{pattern_survey}"))
    too_data_list = sorted(glob.glob(f"{path_to_monitor}/{pattern_too}"))
    test_data_list = sorted(glob.glob(f"{path_to_monitor}/{pattern_test}"))
    raw_data_list = [os.path.abspath(folder) for folder in too_data_list + survey_data_list]

    data_to_process = [os.path.abspath(data) for data in raw_data_list if data not in logtbl['date']]
    del raw_data_list  # 메모리 해제를 위해 사용하지 않는 리스트 삭제
    gc.collect()  # 가비지 컬렉터를 호출하여 메모리 최적화

    if len(data_to_process) > 0:
        print(f"Data to Process: {len(data_to_process)}")
        data_to_monitor = data_to_process[0]
        print(f"="*60)
        print(f"Monitor: {data_to_monitor}")
        print(f"-"*60)
        gpcom = f"python {str(path_run / code)} {obs} {data_to_monitor}"
        t0 = time.time()
        interval0 = 5
        interval = 30
        print(f"Interval: {interval} sec")

        last_size = get_folder_size(data_to_monitor)
        print(f"Initial Size: {format_size(last_size)}")
        check = True
        time.sleep(interval0)
        while check:
            current_size = get_folder_size(data_to_monitor)
            print(f"Current Size: {format_size(current_size)}", end='')

            if current_size == last_size:
                delt = time.time() - t0
                print(f" --> Done! ({delt:.1f} sec)\n")
                print(gpcom)
                result = subprocess.run(gpcom, shell=True)
                if result.returncode == 0:
                    # 성공적으로 실행된 경우 logtbl에 추가하고 저장
                    # new_row = {'date': data_to_monitor}  # 적절히 구성
                    # logtbl.add_row(new_row)
                    # logtbl.write(logfile, format='csv', overwrite=True)
                    # print(f"Updated log table with {data_to_monitor}")
                    check = False
                else:
                    print(f"Error occurred while executing: {gpcom}")
                    # 에러가 발생하면 루프 종료
                    exit(1)
                
                # 처리 후 사용한 변수 메모리 해제
                del data_to_monitor, current_size, last_size
                gc.collect()
            else:
                print(" --> Still Uploading")
                last_size = current_size
                time.sleep(interval)
                
            # 메모리 사용량 최적화를 위해 변수 삭제 및 가비지 컬렉터 호출
            # del current_size
            gc.collect()
    else:
        del data_to_process  # 사용하지 않는 리스트 삭제
        gc.collect()  # 메모리 최적화
        time.sleep(interval_monitor)
