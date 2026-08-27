@echo off
REM v2: includes pre-computed terrain-aware winds
setlocal enabledelayedexpansion

REM Wind grid discretization:
REM   Directions: 0, 10, 20, ..., 350 (36 bins)
REM   Speeds: 5, 15, 25, 35, 45, 55, 65, 75 (8 bins)
REM Pre-computed grids must exist in .\WindGrids\ (run GenerateWindGrids.bat first)

:loop
set /a "windspeed = (%RANDOM% %% 76) + 5"
set /a "winddir = %RANDOM% %% 360"
set /a "moisture = (%RANDOM% %% 60) + 70"
set /a "ignition = %RANDOM% %% 50"

REM Snap wind direction to nearest 10-degree bin
set /a "snapped_dir = (winddir + 5) / 10 * 10"
if !snapped_dir! GEQ 360 set /a "snapped_dir = 0"

REM Snap wind speed to nearest bin (5, 15, 25, 35, 45, 55, 65, 75)
set /a "snapped_spd = (windspeed + 5) / 10 * 10 - 5"
if !snapped_spd! LSS 5 set /a "snapped_spd = 5"
if !snapped_spd! GTR 75 set /a "snapped_spd = 75"

set "SPEED_FILE=.\WindGrids\vel_!snapped_spd!_!snapped_dir!.asc"
set "DIR_FILE=.\WindGrids\ang_!snapped_spd!_!snapped_dir!.asc"

REM Verify wind grid files exist
if not exist "!SPEED_FILE!" (
    echo ERROR: Missing wind grid !SPEED_FILE! - run GenerateWindGrids.bat first
    exit /b 1
)
if not exist "!DIR_FILE!" (
    echo ERROR: Missing wind grid !DIR_FILE! - run GenerateWindGrids.bat first
    exit /b 1
)

(
echo FlamMap-Inputs-File-Version-1
echo.
echo FUEL_MOISTURES_DATA: 1
echo 0 6 7 8 60 90 16
echo.
echo WIND_SPEED: !windspeed!
echo WIND_DIRECTION: !winddir!
echo WIND_SPEED_UNITS: 0
echo GRIDDED_WIND_SPEED_FILE: !SPEED_FILE!
echo GRIDDED_WINDS_DIRECTION_FILE: !DIR_FILE!
echo FOLIAR_MOISTURE_CONTENT: !moisture!
echo CROWN_FIRE_METHOD: Finney
echo NUMBER_PROCESSORS: 4
echo.
echo #SELECTED FLAMMAP OUTPUTS
echo SPREADRATE:
echo #END SELECTED FLAMMAP OUTPUTS
echo.
echo MTT_RESOLUTION: 30
echo MTT_SIM_TIME: 1440
echo MTT_TRAVEL_PATH_INTERVAL: 500
echo MTT_SPOT_PROBABILITY: 0.0
echo MTT_FILL_BARRIERS: 0
) > mtt.input

echo palisades.tif mtt.input .\Ignitions\ignition_!ignition!.shp 0 .\Outputs\mtt 2 > Cmd.txt

..\bin\runmtt Cmd.txt

move .\Outputs\mtt_MTT_ArrivalTime.tif .\Trials\trail_I!ignition!_WS!windspeed!_WD!winddir!_M!moisture!.tif

rd /S /Q Outputs
md Outputs

goto loop
