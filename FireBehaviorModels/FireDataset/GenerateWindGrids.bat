@echo off
REM GenerateWindGrids.bat - Pre-compute terrain-aware wind grids using WindNinja via runflammap
REM Run this ONCE before using GenerateDatasetV2.bat
REM Requires: gdal_translate on PATH (from GDAL/OSGeo4W install)
REM
REM Produces wind grids for:
REM   Directions: 0, 10, 20, ..., 350 (36 bins)
REM   Speeds: 5, 15, 25, 35, 45, 55, 65, 75 mph (8 bins)
REM   Total: 288 direction/speed combinations
REM
REM Output: .\WindGrids\vel_{speed}_{dir}.asc and .\WindGrids\ang_{speed}_{dir}.asc

setlocal enabledelayedexpansion

REM Create output directories
if not exist ".\WindGrids" md WindGrids
if not exist ".\WindGridsTemp" md WindGridsTemp

REM Verify gdal_translate is available
where gdal_translate >nul 2>&1
if errorlevel 1 (
    echo ERROR: gdal_translate not found on PATH.
    echo Install GDAL or OSGeo4W and add to PATH.
    exit /b 1
)

echo ============================================
echo  Generating pre-computed wind grids
echo  36 directions x 8 speeds = 288 grid pairs
echo ============================================
echo.

set COUNT=0
set TOTAL=288

for /L %%D in (0, 10, 350) do (
    for %%S in (5 15 25 35 45 55 65 75) do (
        set /a COUNT+=1
        echo [!COUNT!/%TOTAL%] Generating wind grid: speed=%%S mph, direction=%%D deg

        REM Write FlamMap input file for this wind combo
        (
            echo FlamMap-Inputs-File-Version-1
            echo.
            echo FUEL_MOISTURES_DATA: 1
            echo 0 6 7 8 60 90 16
            echo.
            echo WIND_SPEED: %%S
            echo WIND_DIRECTION: %%D
            echo WIND_SPEED_UNITS: 0
            echo GRIDDED_WINDS_GENERATE: Yes
            echo GRIDDED_WINDS_RESOLUTION: 30
            echo GRIDDED_WINDS_DIURNAL: No
            echo FOLIAR_MOISTURE_CONTENT: 100
            echo CROWN_FIRE_METHOD: Finney
            echo NUMBER_PROCESSORS: 4
            echo.
            echo #SELECTED FLAMMAP OUTPUTS
            echo WINDDIRGRID:
            echo WINDSPEEDGRID:
            echo #END SELECTED FLAMMAP OUTPUTS
        ) > .\WindGridsTemp\flammap_wind.input

        REM Write command file
        echo palisades.tif .\WindGridsTemp\flammap_wind.input .\WindGridsTemp\wind 2 > .\WindGridsTemp\Cmd.txt

        REM Run FlamMap to generate gridded winds
        ..\bin\runflammap .\WindGridsTemp\Cmd.txt

        REM Convert GeoTIFF outputs to ASCII grids
        gdal_translate -of AAIGrid .\WindGridsTemp\wind_WindSpeedGrid.tif .\WindGrids\vel_%%S_%%D.asc
        gdal_translate -of AAIGrid .\WindGridsTemp\wind_WindDirGrid.tif .\WindGrids\ang_%%S_%%D.asc

        REM Clean up temp outputs
        del /Q .\WindGridsTemp\wind_*.tif 2>nul
        del /Q .\WindGridsTemp\wind_*.tif.aux.xml 2>nul
    )
)

REM Clean up temp directory
rd /S /Q WindGridsTemp

echo.
echo ============================================
echo  Done! Generated %TOTAL% wind grid pairs in .\WindGrids\
echo  You can now run GenerateDatasetV2.bat
echo ============================================
