
# def run_scamp_suite(inim, presolve=False):
#     from gppy.utils import lapse

#     factory_dir = (
#         "/data/pipeline_reform/dhhyun_lab/scamptest/test_modular_suite/factory"
#     )
#     os.makedirs(factory_dir, exist_ok=True)

#     lapse("#" * 50 + "start")

#     fname = os.path.basename(inim)
#     soft_link = os.path.join(factory_dir, fname)  # factory/inim.fits
#     if not os.path.exists(soft_link):
#         os.symlink(inim, soft_link)

#     lapse("#" * 50 + "soft link")

#     if presolve:
#         # factory/inim_solved.fits
#         solved_fits = external.solve_field(soft_link, dump_dir=factory_dir)
#     else:
#         # factory/inim.fits
#         solved_fits = soft_link

#     lapse("#" * 50 + "solve-fleid")

#     # pre-SEx
#     external.run_sextractor(solved_fits)  # factory/inim_solved.cat
#     outcat = os.path.splitext(solved_fits)[0] + ".cat"

#     lapse("#" * 50 + "presex")

#     # scamp
#     external.run_scamp(outcat, joint=False)

#     soft_link_head = "_".join(outcat.split("_")[:-1]) + ".head"
#     solved_head = os.path.splitext(outcat)[0] + ".head"
#     os.symlink(solved_head, soft_link_head)  # factory/inim.head

#     lapse("#" * 50 + "scamp")

#     # missfits
#     run_missfits(soft_link)  # makes factory/inim.fits (wcs updated)

#     lapse("#" * 50 + "missfits")


# def _update_polygon_info(calimlist):
#     """later use .head file"""

#     # t0_wcs = time.time()
#     for cc, calim in enumerate(calimlist):
#         # Extract WCS information (center, CD matrix)
#         center, vertices, cd_matrixs = tool.get_wcs_coordinates(calim)
#         cd1_1, cd1_2, cd2_1, cd2_2 = cd_matrixs

#         # updates = [
#         # 	("CTYPE1", 'RA---TPV', 'WCS projection type for this axis'),
#         # 	("CTYPE2", 'DEC--TPV', 'WCS projection type for this axis')
#         # ]
#         # Define header list to udpate
#         updates = [
#             ("RACENT", round(center[0].item(), 3), "RA CENTER [deg]"),
#             ("DECCENT", round(center[1].item(), 3), "DEC CENTER [deg]"),
#         ]

#         # updates.append(("RACENT", round(center[0].item(), 3), "RA CENTER [deg]"))
#         # updates.append(("DECCENT", round(center[1].item(), 3), "DEC CENTER [deg]"))

#         # RA, Dec Polygons
#         for ii, (_ra, _dec) in enumerate(vertices):
#             updates.append((f"RAPOLY{ii}", round(_ra, 3), f"RA POLYGON {ii} [deg]"))
#             updates.append((f"DEPOLY{ii}", round(_dec, 3), f"DEC POLYGON {ii} [deg]"))

#         # Field Rotation
#         try:
#             if (cd1_1 != 0) and (cd1_2 != 0) and (cd2_1 != 0) and (cd2_2 != 0):
#                 rotation_angle_1, rotation_angle_2 = tool.calculate_field_rotation(
#                     cd1_1, cd1_2, cd2_1, cd2_2
#                 )
#             else:
#                 rotation_angle_1, rotation_angle_2 = float("nan"), float("nan")
#         except Exception as e:
#             print(f"Error: {e}")
#             print(f"Image: {calim}")
#             rotation_angle_1, rotation_angle_2 = float("nan"), float("nan")

#         # Update rotation angle
#         updates.append(("ROTANG1", rotation_angle_1, "Rotation angle from North [deg]"))
#         updates.append(("ROTANG2", rotation_angle_2, "Rotation angle from East [deg]"))

#         # FITS header update
#         with fits.open(calim, mode="update") as hdul:
#             for key, value, comment in updates:
#                 hdul[0].header[key] = (value, comment)
#             hdul.flush()  # 변경 사항을 디스크에 저장

# def calculate_field_rotation(cd1_1, cd1_2, cd2_1, cd2_2):
#     """
#     Calculate the field rotation angle based on the given CD matrix elements.
#     The field rotation angles indicate how much the image is rotated with respect
#     to the North and East directions in the celestial coordinate system.

#     Parameters:
#     - cd1_1: CD1_1 value from the FITS header
#     - cd1_2: CD1_2 value from the FITS header
#     - cd2_1: CD2_1 value from the FITS header
#     - cd2_2: CD2_2 value from the FITS header

#     Returns:
#     - rotation_angle_1: The rotation angle of the image's x-axis (typically Right Ascension)
#       from the North in degrees. A positive value indicates a clockwise rotation from North.
#     - rotation_angle_2: The rotation angle of the image's y-axis (typically Declination)
#       from the East in degrees. A positive value indicates a counterclockwise rotation from East.

#     The rotation angles help in understanding how the image is aligned with the celestial coordinate system,
#     which is crucial for accurate star positioning and data alignment in astronomical observations.
#     """
#     rotation_angle_1 = np.degrees(np.arctan(cd1_2 / cd1_1))
#     rotation_angle_2 = np.degrees(np.arctan(cd2_1 / cd2_2))

#     return rotation_angle_1, rotation_angle_2


# def get_wcs_coordinates(filename):
#     # fits 파일을 메모리 매핑을 사용하여 열기
#     with fits.open(filename, memmap=True) as hdulist:
#         header = hdulist[0].header
#         wcs_header = WCS(header)
#         # 데이터의 형태만 가져오기 위해 메모리에 로드하지 않고 헤더 정보 사용
#         data_shape = hdulist[0].shape

#         cd1_1 = header.get("CD1_1", 0)
#         cd1_2 = header.get("CD1_2", 0)
#         cd2_1 = header.get("CD2_1", 0)
#         cd2_2 = header.get("CD2_2", 0)

#     # 중심 좌표 계산
#     center_x = data_shape[1] // 2
#     center_y = data_shape[0] // 2
#     center_ra, center_dec = wcs_header.all_pix2world(center_x, center_y, 0)

#     # 꼭짓점 좌표 계산
#     corners = np.array(
#         [
#             [0, 0],
#             [data_shape[1] - 1, 0],
#             [0, data_shape[0] - 1],
#             [data_shape[1] - 1, data_shape[0] - 1],
#         ]
#     )
#     # 모든 꼭짓점에 대한 월드 좌표를 한 번에 계산
#     world_coordinates = wcs_header.all_pix2world(corners, 0)

#     # 꼭짓점 좌표를 1차원 배열로 변환
#     vertices = world_coordinates

#     # 결과 반환
#     return (center_ra, center_dec), vertices, (cd1_1, cd1_2, cd2_1, cd2_2)


# if __name__ == "__main__":
#     """For a single image"""
#     import sys

#     args = sys.argv
#     run_scamp_suite(args[1])
