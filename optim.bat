@echo off
for /L %%i in (1,1,25) do (
    echo Ausführung %%i
    python start_optim.py
)
pause