@echo off
setlocal enabledelayedexpansion

:: 要关闭的端口列表
set "PORTS=8888 8889"

echo 正在查找并终止占用端口 %PORTS% 的进程...

for %%p in (%PORTS%) do (
    echo.
    echo 检查端口 %%p...
    
    :: 使用 netstat 查找监听在 127.0.0.1:端口 的 PID
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr /R /C:"TCP[ ]*127\.0\.0\.1:%%p "') do (
        set "PID=%%a"
        echo   发现 PID !PID! 占用 127.0.0.1:%%p
        
        :: 终止进程
        taskkill /F /PID !PID! >nul 2>&1
        if !errorlevel! equ 0 (
            echo   已终止 PID !PID!
        ) else (
            echo   无法终止 PID !PID!（可能权限不足或进程已退出）
        )
    )
)

echo.
echo 操作完成。