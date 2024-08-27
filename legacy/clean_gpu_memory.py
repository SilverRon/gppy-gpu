import os

output_file = "gpu_status.txt"

os.system(f"nvidia-smi > {output_file}")

f = open(output_file, "r")
lines = f.readlines()
f.close()

for line in lines[22:]:
	if "python" in line:
		part = line.split(' ')
		part = [val for val in part if val != '']
		pid = part[4]
		# print(pid)
		killcom = f"kill -9 {pid}"
		print(killcom)
		os.system(killcom)