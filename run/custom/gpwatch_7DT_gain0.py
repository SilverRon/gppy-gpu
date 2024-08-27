#============================================================
import os
import glob
import sys
import time
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
    # KB, MB, GB 단위로 변환
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
path_to_obsdata = f"/large_data/obsdata"
path_to_monitor = f"{path_to_obsdata}/{obs}"
path_to_log = f"/large_data/factory/log"
path_to_gppy = f"/home/gp/gppy"
code = f"7DT_Routine_1x1_gain0.py"
#------------------------------------------------------------
while True:
    #------------------------------------------------------------
    #   Log Table
    #------------------------------------------------------------
    logtbl = Table.read(f"{path_to_log}/{obs.lower()}.log", format='csv')
    #------------------------------------------------------------
    #   DATE Folders (=Raw Data)
    #------------------------------------------------------------
    pattern_survey  = "20??-??-??_gain0"
    pattern_too = "20??-??-??_ToO_gain0"
    pattern_test = "20??-00-??"

    survey_data_list = sorted(glob.glob(f"{path_to_monitor}/{pattern_survey}"))
    too_data_list = sorted(glob.glob(f"{path_to_monitor}/{pattern_too}"))
    test_data_list = sorted(glob.glob(f"{path_to_monitor}/{pattern_test}"))
    raw_data_list = too_data_list+survey_data_list
    # for _data in test_data_list: raw_data_list.remove(_data)

    # if len(raw_data_list) > 0:
    #     print(f"Survey Data    : {len(pattern_survey)}")
    #     print(f"ToO Data       : {len(too_data_list)}")
    #     print(f"All Data       : {len(raw_data_list)}")

    data_to_process = [data for data in raw_data_list if data not in logtbl['date']]
    if len(data_to_process) > 0:
        print(f"Data to Process: {len(data_to_process)}")

        # for dd, data in enumerate(data_to_process):
            # print(f"[{dd+1:0>2}] {os.path.basename(data)}")
        #------------------------------------------------------------
        #   Monitor
        #------------------------------------------------------------
        data_to_monitor = data_to_process[0]
        print(f"="*60)
        print(f"Monitor: {data_to_monitor}")
        print(f"-"*60)
        gpcom = f"python {path_to_gppy}/{code} {obs} {data_to_monitor}"
        t0 = time.time()
        interval0 = 5
        interval = 30
        print(f"Interval: {interval} sec")

        last_size = get_folder_size(data_to_monitor)
        print(f"Initial Size: {format_size(last_size)}",)
        check = True
        time.sleep(interval0)
        while check:
            current_size = get_folder_size(data_to_monitor)
            print(f"Current Size: {format_size(current_size)}", end='')

            if current_size == last_size:
                delt = time.time() - t0
                print(f" --> Done! ({delt:.1f} sec)\n")
                print(gpcom)
                os.system(gpcom)
                check = False
                # break
            else:
                print(" --> Still Uploading")

            last_size = current_size
            time.sleep(interval)
    else:
        time.sleep(interval_monitor)
