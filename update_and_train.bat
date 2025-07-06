@echo off
chcp 65001 > nul
 
echo --------------------------------------------------
echo 正在啟動資料更新與模型訓練批次檔...
echo 當前目錄: %cd%
echo --------------------------------------------------

REM 進入專案根目錄
cd /d "F:\python"
echo 已切換到專案根目錄: %cd%

REM 獲取 Python 解釋器的路徑
REM 您需要將 "C:\Python313\python.exe" 替換為您系統上 Python 解釋器的實際路徑
REM 如果您不確定，可以在命令提示字元中輸入 'where python' 來查找
set PYTHON_EXE="C:\Python313\python.exe"
echo 設定的 Python 解釋器路徑: %PYTHON_EXE%

echo.
echo --------------------------------------------------
echo 正在執行資料更新腳本 (data_updater.py)...
echo 命令: %PYTHON_EXE% "data\data_updater.py"
echo --------------------------------------------------
%PYTHON_EXE% "data\data_updater.py"

if %errorlevel% neq 0 (
    echo.
    echo --------------------------------------------------
    echo 錯誤：資料更新腳本執行失敗！錯誤碼: %errorlevel%
    echo --------------------------------------------------
    pause
    goto :eof
)

echo.
echo --------------------------------------------------
echo 資料更新完成
echo 正在執行模型訓練腳本 (train.py)...
echo 命令: %PYTHON_EXE% "scripts\train.py"
echo --------------------------------------------------
%PYTHON_EXE% "scripts\train.py"

if %errorlevel% neq 0 (
    echo.
    echo --------------------------------------------------
    echo 錯誤：模型訓練腳本執行失敗！錯誤碼: %errorlevel%
    echo --------------------------------------------------
    pause
    goto :eof
)

echo.
echo --------------------------------------------------
echo 模型訓練完成。
echo 所有任務已完成。
echo --------------------------------------------------
