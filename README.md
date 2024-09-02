# gppy-gpu
Automatic imaging processing pipeline using GPU for 7-Dimensional Telescope

dhhyun ChangeLog

Source files such as phot and util packages were moved into gppy-gpu/src.
Runtime files like 7DT_Routine_1x1_gain2750.py and gpwatch_7DT_gain2750.py were repositioned into gppy-gpu/run/routine. Other custom runs such as gain 0 or dedicated analyses are supposed to be placed in gppy-gpu/run/custom.

Paths to change: path_base