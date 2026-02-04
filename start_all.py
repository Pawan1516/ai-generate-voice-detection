"""
Start both Frontend (port 3000) and Backend (port 8000) together
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def main():
    print("""
╔════════════════════════════════════════════════════════════════╗
║     🚀 AI VOICE DETECTION - FULL STACK STARTUP               ║
╚════════════════════════════════════════════════════════════════╝
    """)
    
    base_dir = Path(__file__).parent
    backend_dir = base_dir / 'backend'
    frontend_dir = base_dir / 'frontend'
    
    # Check if npm is installed
    print("📋 Checking prerequisites...")
    try:
        subprocess.run(['npm', '--version'], capture_output=True, check=True)
        print("   ✓ Node.js/npm found")
    except:
        print("   ❌ Node.js/npm not found!")
        print("   Install from: https://nodejs.org/")
        return
    
    # Install frontend dependencies if needed
    print("\n📦 Installing frontend dependencies...")
    if not (frontend_dir / 'node_modules').exists():
        print("   Installing npm packages...")
        subprocess.run(['npm', 'install'], cwd=frontend_dir)
        print("   ✓ Dependencies installed")
    else:
        print("   ✓ Dependencies already installed")
    
    # Start backend
    print("\n🔧 Starting Backend Server (port 8000)...")
    print("   Command: python -m uvicorn main:app --host 127.0.0.1 --port 8000")
    
    backend_process = subprocess.Popen(
        [sys.executable, '-m', 'uvicorn', 'main:app', 
         '--host', '127.0.0.1', '--port', '8000', '--reload'],
        cwd=backend_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    time.sleep(3)  # Wait for backend to start
    
    # Start frontend
    print("\n🎨 Starting Frontend Server (port 3000)...")
    print("   Command: node server.js")
    
    frontend_process = subprocess.Popen(
        ['node', 'server.js'],
        cwd=frontend_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    time.sleep(2)  # Wait for frontend to start
    
    print("""
╔════════════════════════════════════════════════════════════════╗
║                    ✅ SERVERS RUNNING!                        ║
╚════════════════════════════════════════════════════════════════╝

🌐 Frontend:  http://localhost:3000
🔌 Backend:   http://localhost:8000
📚 API Docs:  http://localhost:8000/docs

Press CTRL+C to stop all servers
    """)
    
    try:
        # Keep both processes running
        while True:
            if backend_process.poll() is not None:
                print("❌ Backend crashed!")
                break
            if frontend_process.poll() is not None:
                print("❌ Frontend crashed!")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n⛔ Shutting down servers...")
        backend_process.terminate()
        frontend_process.terminate()
        time.sleep(1)
        backend_process.kill()
        frontend_process.kill()
        print("✅ All servers stopped")

if __name__ == '__main__':
    main()
