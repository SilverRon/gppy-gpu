# gppy-gpu
Automatic imaging processing pipeline using GPU for 7-Dimensional Telescope

dhhyun ChangeLog

Source files such as phot and util packages were moved into gppy-gpu/src.
Runtime files like 7DT_Routine_1x1_gain2750.py and gpwatch_7DT_gain2750.py were repositioned into gppy-gpu/run/routine. Other custom runs such as gain 0 or dedicated analyses are supposed to be placed in gppy-gpu/run/custom.

Now major paths are managed by a file named "path.json".
This file should be placed in the same directory as gpwatch or 7DT_Routine and the code wouldn't run without it.
"path_base" is a required input whereas other entries can be empty if you want to use the default path structure.

Additionally, all running files -- many in "legacy" -- other than those in run/routine are no longer maintained.
Files in run/custom are there for demonstration, and would not work now due to changed paths.

