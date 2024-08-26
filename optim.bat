@echo off
for /L %%i in (1,1,15) do (
    echo Ausführung %%i
    python start_optim.py
)
pause