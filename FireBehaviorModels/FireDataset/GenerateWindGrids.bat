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

REM Use the script's own directory as base
set "BASEDIR=%~dp0"
set "TEMPDIR=%BASEDIR%WindGridsTemp"
set "OUTDIR=%BASEDIR%WindGrids"

REM Create output directories
if not exist "%TEMPDIR%" md "%TEMPDIR%"
if not exist "%OUTDIR%" md "%OUTDIR%"

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

        call :WriteInput %%S %%D

        REM Write command file
        echo palisades.tif %TEMPDIR%\flammap_wind.input %TEMPDIR%\wind 2> "%TEMPDIR%\Cmd.txt"

        REM Run FlamMap to generate gridded winds
        ..\bin\runflammap "%TEMPDIR%\Cmd.txt"

        REM Convert GeoTIFF outputs to ASCII grids
        gdal_translate -of AAIGrid "%TEMPDIR%\wind_WindSpeedGrid.tif" "%OUTDIR%\vel_%%S_%%D.asc"
        gdal_translate -of AAIGrid "%TEMPDIR%\wind_WindDirGrid.tif" "%OUTDIR%\ang_%%S_%%D.asc"

        REM Clean up temp outputs
        del /Q "%TEMPDIR%\wind_*.tif" 2>nul
        del /Q "%TEMPDIR%\wind_*.tif.aux.xml" 2>nul
    )
)

REM Clean up temp directory
rd /S /Q "%TEMPDIR%"

echo.
echo ============================================
echo  Done! Generated %TOTAL% wind grid pairs in .\WindGrids\
echo  You can now run GenerateDatasetV2.bat
echo ============================================

exit /b 0

:WriteInput
REM Subroutine to write the FlamMap input file
REM %1 = wind speed, %2 = wind direction
set "INPUTFILE=%TEMPDIR%\flammap_wind.input"
echo FlamMap-Inputs-File-Version-1> "%INPUTFILE%"
echo.>> "%INPUTFILE%"
echo FUEL_MOISTURES_DATA: 1>> "%INPUTFILE%"
echo 0 6 7 8 60 90 16>> "%INPUTFILE%"
echo.>> "%INPUTFILE%"
echo WIND_SPEED: %1>> "%INPUTFILE%"
echo WIND_DIRECTION: %2>> "%INPUTFILE%"
echo WIND_SPEED_UNITS: 0>> "%INPUTFILE%"
echo GRIDDED_WINDS_GENERATE: Yes>> "%INPUTFILE%"
echo GRIDDED_WINDS_RESOLUTION: 30>> "%INPUTFILE%"
echo GRIDDED_WINDS_DIURNAL: No>> "%INPUTFILE%"
echo FOLIAR_MOISTURE_CONTENT: 100>> "%INPUTFILE%"
echo CROWN_FIRE_METHOD: Finney>> "%INPUTFILE%"
echo NUMBER_PROCESSORS: 4>> "%INPUTFILE%"
echo.>> "%INPUTFILE%"
echo #SELECTED FLAMMAP OUTPUTS>> "%INPUTFILE%"
echo WINDDIRGRID:>> "%INPUTFILE%"
echo WINDSPEEDGRID:>> "%INPUTFILE%"
echo #END SELECTED FLAMMAP OUTPUTS>> "%INPUTFILE%"
exit /b 0
