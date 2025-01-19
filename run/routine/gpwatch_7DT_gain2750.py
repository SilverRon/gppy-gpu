#============================================================
import os
import subprocess
import glob
import sys
import time
import json
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
obs = get_input().upper()
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
    raw_data_list = [os.path.abspath(folder) for folder in too_data_list+survey_data_list]
    # for _data in test_data_list: raw_data_list.remove(_data)

    # if len(raw_data_list) > 0:
    #     print(f"Survey Data    : {len(pattern_survey)}")
    #     print(f"ToO Data       : {len(too_data_list)}")
    #     print(f"All Data       : {len(raw_data_list)}")

    data_to_process = [os.path.abspath(data) for data in raw_data_list if data not in logtbl['date']]
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
        gpcom = f"python {str(path_run / code)} {obs} {data_to_monitor}"
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
                # print(gpcom)
                # os.system(gpcom)
                #   Run Pipeline
                try:
                    print(f"Run Pipeline: {gpcom}")
                    result = subprocess.run(gpcom.split(), check=True, capture_output=True, text=True)
                    print(f"Pipeline executed successfully:\n{result.stdout}")
                except subprocess.CalledProcessError as e:
                    print(f"Error occurred while executing pipeline:\n{e.stderr}")
                except KeyboardInterrupt:
                    print("Process interrupted by user. Exiting safely...")
                    sys.exit(1)  # 안전한 종료
                except Exception as e:
                    print(f"Unexpected error: {str(e)}")
                check = False
                # break
            else:
                print(" --> Still Uploading")

            last_size = current_size
            time.sleep(interval)
    else:
        time.sleep(interval_monitor)
