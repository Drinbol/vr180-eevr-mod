echo off
:::::::::::::::::::::::
:: path to blender path
:::::::::::::::::::::::
SET "blenderPath=D:\Games\Steam\steamapps\common\Blender\"
SET "eevrPath=E:\Project\mmd\vrcam\VRRenderer_CUBE_TO_EQU_ver_05.py"
::::::::::::::::::::
::  input parameter 
::::::::::::::::::::
SET /P startFrame= start frame:
SET /P endFrame= end frame:
SET /P sceneFile= scene file:

SET /A startFrame2=%startFrame%+1
SET /A startFrame3=%startFrame%+2
SET /A test=0

echo on
start /d "%blenderPath%" blender.exe -b "%cd%\%sceneFile%" -P "%eevrPath%" -- --FOV 180 --startFrame %startFrame% --endFrame %endFrame% --step 2 --test %test%
timeout 1
start /d "%blenderPath%" blender.exe -b "%cd%\%sceneFile%" -P "%eevrPath%" -- --FOV 180 --startFrame %startFrame2% --endFrame %endFrame% --step 2 --test %test%
timeout 100

