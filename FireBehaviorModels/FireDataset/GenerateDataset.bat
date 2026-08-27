@echo off
setlocal enabledelayedexpansion

:loop
set /a "windspeed = (%RANDOM% %% 76) + 5"
set /a "winddir = %RANDOM% %% 360"
set /a "moisture = (%RANDOM% %% 60) + 70"
set /a "ignition = %RANDOM% %% 50"

(
echo ShortTerm-Inputs-File-Version-1
echo.
echo FUEL_MOISTURES_DATA: 1
echo 0 6 7 8 60 90 16
echo.
echo WIND_SPEED: !windspeed!
echo WIND_DIRECTION: !winddir!
echo FOLIAR_MOISTURE_CONTENT: !moisture!
echo CROWN_FIRE_METHOD: Finney
echo NUMBER_PROCESSORS: 4
echo.
echo MTT_RESOLUTION: 30
echo MTT_SIM_TIME: 1440
echo MTT_TRAVEL_PATH_INTERVAL: 500
echo MTT_SPOT_PROBABILITY: 0.0
echo.
echo MTT_FILL_BARRIERS: 0
) > mtt.input

echo palisades.tif mtt.input .\Ignitions\ignition_!ignition!.shp 0 .\Outputs\mtt 2 > Cmd.txt

..\bin\TestMTT Cmd.txt

move .\Outputs\mtt_MTT_ArrivalTime.tif .\Trials\trail_I!ignition!_WS!windspeed!_WD!winddir!_M!moisture!.tif

rd /S /Q Outputs
md Outputs

goto loop