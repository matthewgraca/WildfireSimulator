@echo off

set FM_ROOT=%~dp0
set FM_ROOT=%FM_ROOT:\\=\%
@echo Setting environment for using FireBehaviorModels.

echo %FM_ROOT%

SET "PATH=%FM_ROOT%bin;%PATH%"
SET "GDAL_DATA=%FM_ROOT%bin\share\gdal-data"
SET "PROJ_LIB=%FM_ROOT%bin\share\proj"
SET "WINDNINJA_DATA=%FM_ROOT%bin\share\windninja-data"

