@echo off
REM Start both Frontend (3000) and Backend (8000)
REM This is a Windows batch script

echo.
echo ========================================
echo  AI Voice Detection - Full Stack
echo ========================================
echo.
echo Starting frontend and backend...
echo.

REM Check if npm is installed
npm --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js/npm is not installed!
    echo Install from: https://nodejs.org/
    pause
    exit /b 1
)

REM Install frontend dependencies if needed
if not exist "frontend\node_modules" (
    echo Installing frontend dependencies...
    cd frontend
    call npm install
    cd ..
)

REM Start both servers
echo.
echo Starting servers...
echo   Frontend: http://localhost:3000
echo   Backend:  http://localhost:8000
echo.

REM Open in browser (optional)
start http://localhost:3000

REM Start frontend in background
start "AI Voice Detection - Frontend" cmd /k "cd frontend && node server.js"

REM Wait a moment for frontend to start
timeout /t 2 /nobreak

REM Start backend
start "AI Voice Detection - Backend" cmd /k "cd backend && python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload"

echo.
echo ========================================
echo  Servers Started!
echo ========================================
echo.
echo Frontend:  http://localhost:3000
echo Backend:   http://localhost:8000
echo API Docs:  http://localhost:8000/docs
echo.
echo Close the windows to stop servers
echo ========================================
echo.
pause
