@echo off
if "%1" == "" (
    echo Usage run.bat ^<flags^>
    echo flags:
    echo -gpu run the gpu version 
    echo -cpu run the cpu version
    echo -cmp compare the output of gpu version and cpu version
) else if "%1" == "gpu" (
    cd bin
    "main_gpu.exe" "../data/params/evaluate.params" "../data/obstacles/none.dat" "../result"
) else if "%1" == "cpu" (
    cd bin
    "main_cpu.exe" "../data/params/evaluate.params" "../data/obstacles/none.dat" "../result"
)
