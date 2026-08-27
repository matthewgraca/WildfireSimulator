@echo off
REM v2: includes pre-computed terrain-aware winds
REM Uses WindNinja (via GRIDDED_WINDS_GENERATE) for terrain-respecting wind fields.
REM Saves WINDDIRGRID and WINDSPEEDGRID TIFs alongside each trial for ML input.
REM Run from the FireDataset directory. Run SetEnv.bat first.
setlocal enabledelayedexpansion

:loop
set /a "windspeed = (%RANDOM% %% 76) + 5"
set /a "winddir = %RANDOM% %% 360"
set /a "moisture = (%RANDOM% %% 60) + 70"
set /a "ignition = %RANDOM% %% 50"

call :WriteInput !windspeed! !winddir! !moisture!

>Cmd.txt echo palisades.tif mtt.input .\Ignitions\ignition_!ignition!.shp 0 .\Outputs\mtt 2

..\bin\runmtt Cmd.txt

set "TRIAL_BASE=trial_I!ignition!_WS!windspeed!_WD!winddir!_M!moisture!"

move .\Outputs\mtt_MTT_ArrivalTime.tif ".\Trials\!TRIAL_BASE!.tif"
move .\Outputs\mtt_WINDSPEED.tif ".\Trials\!TRIAL_BASE!_windspeed.tif"
move .\Outputs\mtt_WINDDIR.tif ".\Trials\!TRIAL_BASE!_winddir.tif"

rd /S /Q Outputs
md Outputs

goto loop

:WriteInput
REM %1 = wind speed, %2 = wind direction, %3 = moisture
set "INPUTFILE=mtt.input"
>"%INPUTFILE%" echo ShortTerm-Inputs-File-Version-1
>>"%INPUTFILE%" echo.
>>"%INPUTFILE%" echo FUEL_MOISTURES_DATA: 1
>>"%INPUTFILE%" echo 0 6 7 8 60 90 16
>>"%INPUTFILE%" echo.
>>"%INPUTFILE%" echo WIND_SPEED: %1
>>"%INPUTFILE%" echo WIND_DIRECTION: %2
>>"%INPUTFILE%" echo WIND_SPEED_UNITS: 0
>>"%INPUTFILE%" echo GRIDDED_WINDS_GENERATE: Yes
>>"%INPUTFILE%" echo GRIDDED_WINDS_RESOLUTION: 30
>>"%INPUTFILE%" echo GRIDDED_WINDS_DIURNAL: No
>>"%INPUTFILE%" echo FOLIAR_MOISTURE_CONTENT: %3
>>"%INPUTFILE%" echo CROWN_FIRE_METHOD: Finney
>>"%INPUTFILE%" echo NUMBER_PROCESSORS: 4
>>"%INPUTFILE%" echo.
>>"%INPUTFILE%" echo MTT_RESOLUTION: 30
>>"%INPUTFILE%" echo MTT_SIM_TIME: 1440
>>"%INPUTFILE%" echo MTT_TRAVEL_PATH_INTERVAL: 500
>>"%INPUTFILE%" echo MTT_SPOT_PROBABILITY: 0.0
>>"%INPUTFILE%" echo MTT_FILL_BARRIERS: 0
exit /b 0
