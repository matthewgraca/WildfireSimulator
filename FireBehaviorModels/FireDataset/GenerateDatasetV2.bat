@echo off
REM v2: includes terrain-aware winds (WindNinja via GRIDDED_WINDS_GENERATE)
REM Saves WINDSPEED and WINDDIR grid TIFs alongside each trial for ML input.
REM
REM Usage:  GenerateDatasetV2.bat <landscape_name>
REM   e.g.  GenerateDatasetV2.bat palisades
REM         GenerateDatasetV2.bat angeles
REM
REM Expects the following layout under .\landscapes\<name>\ :
REM   <name>.tif        the landscape LCP GeoTIFF
REM   Ignitions\        ignition_0.shp ... ignition_N.shp
REM   Trials\           (output) arrival-time + wind grid TIFs
REM   Outputs\          (scratch) cleared each iteration
REM
REM Run SetEnv.bat (in FireBehaviorModels) first to set GDAL/PROJ/WindNinja paths.
setlocal enabledelayedexpansion

REM --- Resolve landscape name argument ---
if "%~1"=="" (
    echo ERROR: no landscape name given.
    echo Usage: GenerateDatasetV2.bat ^<landscape_name^>
    echo Example: GenerateDatasetV2.bat angeles
    exit /b 1
)
set "LSNAME=%~1"
set "LSDIR=.\landscapes\%LSNAME%"
set "LANDSCAPE=%LSDIR%\%LSNAME%.tif"

if not exist "%LANDSCAPE%" (
    echo ERROR: landscape file not found: %LANDSCAPE%
    exit /b 1
)
if not exist "%LSDIR%\Ignitions" (
    echo ERROR: ignitions folder not found: %LSDIR%\Ignitions
    exit /b 1
)

if not exist "%LSDIR%\Trials" md "%LSDIR%\Trials"
if not exist "%LSDIR%\Outputs" md "%LSDIR%\Outputs"

echo Generating trials for landscape: %LSNAME%
echo Landscape file: %LANDSCAPE%
echo.

:loop
set /a "windspeed = (%RANDOM% %% 76) + 5"
set /a "winddir = %RANDOM% %% 360"
set /a "moisture = (%RANDOM% %% 60) + 70"
set /a "ignition = %RANDOM% %% 50"

call :WriteInput !windspeed! !winddir! !moisture!

>"%LSDIR%\Cmd.txt" echo %LANDSCAPE% %LSDIR%\mtt.input %LSDIR%\Ignitions\ignition_!ignition!.shp 0 %LSDIR%\Outputs\mtt 2

..\bin\runmtt "%LSDIR%\Cmd.txt"

set "TRIAL_BASE=trial_I!ignition!_WS!windspeed!_WD!winddir!_M!moisture!"

if exist "%LSDIR%\Outputs\mtt_MTT_ArrivalTime.tif" (
    move "%LSDIR%\Outputs\mtt_MTT_ArrivalTime.tif" "%LSDIR%\Trials\!TRIAL_BASE!.tif"
    move "%LSDIR%\Outputs\mtt_WINDSPEED.tif" "%LSDIR%\Trials\!TRIAL_BASE!_windspeed.tif"
    move "%LSDIR%\Outputs\mtt_WINDDIR.tif" "%LSDIR%\Trials\!TRIAL_BASE!_winddir.tif"
) else (
    echo No spread for I!ignition! WS!windspeed! WD!winddir! M!moisture! - skipping
)

rd /S /Q "%LSDIR%\Outputs"
md "%LSDIR%\Outputs"

goto loop

:WriteInput
REM %1 = wind speed, %2 = wind direction, %3 = moisture
set "INPUTFILE=%LSDIR%\mtt.input"
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
>>"%INPUTFILE%" echo NUMBER_PROCESSORS: 8
>>"%INPUTFILE%" echo.
>>"%INPUTFILE%" echo MTT_RESOLUTION: 30
>>"%INPUTFILE%" echo MTT_SIM_TIME: 1440
>>"%INPUTFILE%" echo MTT_TRAVEL_PATH_INTERVAL: 500
>>"%INPUTFILE%" echo MTT_SPOT_PROBABILITY: 0.0
>>"%INPUTFILE%" echo MTT_FILL_BARRIERS: 0
exit /b 0
