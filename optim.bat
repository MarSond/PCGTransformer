@echo off
for /L %%i in (1,1,50) do (
    echo Ausführung %%i
    python start_optim_embedding.py
)
pause