# %%
# Python Library
import os, glob, sys
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from astropy.stats import sigma_clip
# import cupy as cp
from astropy.io import fits
import warnings
warnings.filterwarnings("ignore")

# Plot presetting
import matplotlib.pyplot as plt
import matplotlib as mpl
#
mpl.rcParams["axes.titlesize"] = 14
mpl.rcParams["axes.labelsize"] = 20
plt.rcParams['savefig.dpi'] = 500
plt.rc('font', family='serif')

# %%
def smooth_array(arr, smoothing_size):
    """
    이 함수는 넘파이 배열을 입력으로 받고, 주어진 smoothing size에 따라 이동 평균을 적용하여 스무딩된 배열을 반환합니다.
    
    :param arr: 스무딩할 원본 넘파이 배열
    :param smoothing_size: 사용할 이동 평균의 크기 (양의 정수)
    :return: 스무딩된 넘파이 배열
    """
    if smoothing_size < 1:
        raise ValueError("Smoothing size는 1 이상이어야 합니다.")
    
    smoothed_arr = np.copy(arr)
    for i in range(smoothing_size, len(arr) - smoothing_size):
        smoothed_arr[i] = np.mean(arr[i - smoothing_size:i + smoothing_size + 1])

    return smoothed_arr

def sigma_clipping_gpu(data, sigma=3):
    # Astropy의 sigma_clip을 사용하여 NumPy 배열에서 클리핑 수행
    clipped_data_numpy = sigma_clip(cp.asnumpy(data), sigma=sigma, cenfunc='median')

    # NumPy 배열을 CuPy 배열로 변환
    clipped_data = cp.asarray(clipped_data_numpy)

    # NaN이나 무한대 값을 중간값으로 대체
    median_value = cp.median(data[cp.isfinite(data)])
    clipped_data[~cp.isfinite(clipped_data)] = median_value

    return clipped_data

def evaluate_flatness(data):
    """
    이 함수는 넘파이 배열에 대한 선형 회귀를 수행하여 이미지의 평평함을 평가합니다.
    
    :param data: 선형 회귀를 수행할 데이터 배열
    :return: 기울기, 절편, 상관 계수, p-value, 표준 오차
    """
    x = np.arange(len(data))
    slope, intercept, r_value, p_value, std_err = linregress(x, data)
    return slope, intercept, r_value, p_value, std_err

def sigma_clipping(data, sigma=3):
    """
    이 함수는 sigma clipping을 사용하여 이미지 데이터에서 이상치를 제거합니다.
    
    :param data: 분석할 이미지 데이터
    :param sigma: 클리핑할 시그마 수준
    :return: 클리핑된 이미지 데이터
    """
    clipped_data = sigma_clip(data, sigma=sigma, cenfunc='median')
    return clipped_data.filled(np.median(data))

def sigma_clipping_gpu(data, sigma=3):
    # 이 부분은 astropy의 sigma_clip을 CuPy 배열에 맞게 구현해야 할 수 있습니다.
    # 예시 구현은 아래와 같을 수 있습니다.
    clipped_data = cp.asarray(sigma_clip(cp.asnumpy(data), sigma=sigma, cenfunc='median'))
    return clipped_data.filled(cp.median(data))

def read_partial_fits(file_path, x_slice, y_slice):
    """
    FITS 파일에서 지정된 영역의 데이터만 읽어들이는 함수.
    
    :param file_path: 읽을 FITS 파일의 경로
    :param x_slice: 데이터를 읽을 x축 범위 (시작, 끝)
    :param y_slice: 데이터를 읽을 y축 범위 (시작, 끝)
    :return: 지정된 영역의 데이터
    """
    with fits.open(file_path, memmap=True) as hdul:
        data = hdul[0].data[y_slice[0]:y_slice[1], x_slice[0]:x_slice[1]]
    return data

def calculate_mode(data, start=0, end=100, bin_width=0.5):
    # 데이터를 1차원 배열로 변환
    flattened_data = data.flatten()

    # 히스토그램 생성 (여기서 bin의 간격을 정의)
    bin_counts, bin_edges = np.histogram(flattened_data, bins=np.arange(start, end, bin_width))

    # 가장 높은 빈도를 가진 bin 찾기
    max_count_index = np.argmax(bin_counts)
    most_frequent_bin = (bin_edges[max_count_index], bin_edges[max_count_index + 1])

    # print(f"가장 빈번한 bin 범위: {most_frequent_bin}")
    # print(f"해당 bin의 빈도: {bin_counts[max_count_index]}")
    return np.mean(most_frequent_bin)
def make_snapshot(image_path):
    """
    주어진 이미지 파일 경로에 대해 이미지 데이터를 분석하고 결과를 시각화하는 함수.
    
    :param image_path: 분석할 이미지 파일의 경로
    """
    # 이미지 데이터 로드
    outname = os.path.splitext(image_path)[0] + '_snapshot.png'
    if not os.path.exists(outname):
        data, hdr = fits.getdata(image_path, header=True)
        print(f"Generate {outname}")

        # 플롯 생성
        plt.figure(figsize=(7, 5))
        plt.title(os.path.basename(image_path), fontsize=10)
        if 'SKYVAL' in hdr.keys():
            skyval = hdr['SKYVAL']
        else:
            # skyval = calculate_mode(data, start=0, end=100, bin_width=0.5)
            skyval = calculate_mode(data, start=450, end=550, bin_width=2.5)
        plt.imshow(data-skyval, vmin=0, vmax=7.5)
            
        plt.tight_layout()
        # 결과 저장
        plt.savefig(outname, dpi=100)
        plt.close()
    # return slope_sline, intercept_sline, total_change_y


from multiprocessing import Pool

def process_images(image_paths, num_processes=4):
    """
    여러 이미지 파일에 대해 make_snapshot 함수를 병렬로 실행합니다. 병렬 처리에 사용할 프로세스 수를 설정합니다.

    :param image_paths: 분석할 이미지 파일 경로의 리스트
    :param num_processes: 사용할 프로세스 수
    """
    with Pool(processes=num_processes) as pool:
        pool.map(make_snapshot, image_paths)

obj = input(f"Type Field Name (UDS):")
path_proc = '/large_data/processed'

t0 = time.time()

imlist = sorted(glob.glob(f'{path_proc}/{obj}/7DT??/*/calib*0.fits'))

process_images(imlist, num_processes=30)

delt = time.time() - t0
print(f"{delt:.3f} second")
# print(delt, len(imlist))



