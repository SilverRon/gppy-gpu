import os, sys
from astropy.time import Time
from astropy.table import Table
from datetime import datetime


def extract_date_and_time(date_obs_str, round_seconds=False):
    """
    Extract date and time from the 'DATE-OBS' FITS header keyword value.

    Parameters:
    date_obs_str (str): The DATE-OBS string, usually in the format 'YYYY-MM-DDTHH:MM:SS.sss'
    round_seconds (bool): Whether to round the seconds to the nearest whole number

    Returns:
    str, str: Extracted date and time strings in 'YYYYMMDD' and 'HHMMSS' formats
    """
    # Convert the DATE-OBS string to an Astropy Time object
    time_obj = Time(date_obs_str)

    # Extract the date and time components
    date_str = time_obj.strftime("%Y%m%d")
    if round_seconds:
        time_str = time_obj.strftime("%H%M%S")
    else:
        time_str = f"{time_obj.datetime.hour:02}{time_obj.datetime.minute:02}{int(time_obj.datetime.second):02}"

    return date_str, time_str


def calc_mean_dateloc(dateloclist):

    # 문자열을 datetime 객체로 변환
    datetime_objects = [datetime.fromisoformat(t) for t in dateloclist]

    # datetime 객체를 POSIX 시간으로 변환
    posix_times = [dt.timestamp() for dt in datetime_objects]

    # 평균 POSIX 시간 계산
    mean_posix_time = np.mean(posix_times)

    # 평균 POSIX 시간을 datetime 객체로 변환
    mean_datetime = datetime.fromtimestamp(mean_posix_time)

    # 필요한 경우, datetime 객체를 ISOT 형식의 문자열로 변환
    mean_isot_time = mean_datetime.isoformat()
    return mean_isot_time


def inputlist_parser(imagelist_file):
    if os.path.exists(imagelist_file):
        print(f"{imagelist_file} found!")
    else:
        print(f"Not Found {imagelist_file}!")
        sys.exit()
    input_table = Table.read(imagelist_file, format="ascii")
    # input_table = Table.read(imagelist_file_to_stack, format="ascii.commented_header")
    _files = [f for f in input_table["file"].data]
    return _files


def unpack(packed, type, ex=None):
    if len(packed) != 1:
        print(f"There are more than one ({len(packed)}) {type}s")
        unpacked = input(
            f"Type {type.upper()} name (e.g. {packed if ex is None else ex}):"
        )
    else:
        unpacked = packed[0]
    return unpacked
    # return float(unpacked)
