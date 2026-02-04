#!/usr/bin/env python3
"""
Verify that the AI Voice Detection project is properly set up
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_file(path, description):
    """Check if a file exists"""
    if os.path.exists(path):
        print(f"✓ {description}")
        return True
    else:
        print(f"✗ {description} - MISSING")
        return False

def check_command(cmd, description):
    """Check if a command is available"""
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"✓ {description}")
        return True
    except:
        print(f"✗ {description} - NOT INSTALLED")
        return False

def main():
    print("""
╔════════════════════════════════════════════════════════════════╗
║     AI VOICE DETECTION - PROJECT VERIFICATION                ║
╚════════════════════════════════════════════════════════════════╝
    """)
    
    base_dir = Path(__file__).parent
    checks_passed = 0
    checks_total = 0
    
    # Check Python
    print("\n🐍 Python Environment:")
    checks_total += 1
    if check_command([sys.executable, '--version'], "Python installed"):
        checks_passed += 1
    
    # Check Node.js
    print("\n📦 Node.js/npm:")
    checks_total += 1
    if check_command(['npm', '--version'], "npm installed"):
        checks_passed += 1
    
    # Check key files
    print("\n📁 Backend Files:")
    backend_files = [
        ('backend/main.py', 'Backend API (main.py)'),
        ('backend/audio_processor.py', 'Audio processor'),
        ('backend/ml_model.py', 'ML model handler'),
        ('backend/config.py', 'Configuration'),
        ('backend/requirements.txt', 'Python dependencies'),
        ('backend/models/voice_detection_model.pkl', 'Trained model'),
    ]
    
    for file_path, description in backend_files:
        full_path = base_dir / file_path
        checks_total += 1
        if check_file(full_path, description):
            checks_passed += 1
    
    print("\n🎨 Frontend Files:")
    frontend_files = [
        ('frontend/server.js', 'Express server'),
        ('frontend/index.html', 'Main HTML'),
        ('frontend/app.js', 'Frontend logic'),
        ('frontend/config.js', 'Configuration'),
        ('frontend/style.css', 'Styling'),
        ('frontend/package.json', 'npm configuration'),
    ]
    
    for file_path, description in frontend_files:
        full_path = base_dir / file_path
        checks_total += 1
        if check_file(full_path, description):
            checks_passed += 1
    
    print("\n🚀 Startup Scripts:")
    startup_files = [
        ('start_all.py', 'Python startup script'),
        ('start_all.bat', 'Windows batch startup'),
        ('LOCAL_SETUP.md', 'Local setup guide'),
    ]
    
    for file_path, description in startup_files:
        full_path = base_dir / file_path
        checks_total += 1
        if check_file(full_path, description):
            checks_passed += 1
    
    # Check frontend dependencies
    print("\n📚 Frontend Dependencies:")
    node_modules = base_dir / 'frontend' / 'node_modules'
    checks_total += 1
    if os.path.exists(node_modules):
        print("✓ npm dependencies installed")
        checks_passed += 1
    else:
        print("✗ npm dependencies NOT installed - Run: cd frontend && npm install")
    
    # Check backend dependencies
    print("\n🔧 Backend Dependencies:")
    checks_total += 1
    try:
        subprocess.run(
            [sys.executable, '-c', 'import fastapi, uvicorn, librosa, sklearn, numpy'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        print("✓ Python dependencies installed")
        checks_passed += 1
    except:
        print("✗ Python dependencies NOT installed - Run: cd backend && pip install -r requirements.txt")
    
    # Summary
    print(f"""
╔════════════════════════════════════════════════════════════════╗
║                       VERIFICATION SUMMARY                    ║
╚════════════════════════════════════════════════════════════════╝

Checks Passed: {checks_passed}/{checks_total}
    """)
    
    if checks_passed == checks_total:
        print("✅ All checks passed! Ready to run:")
        print("""
    Option 1: python start_all.py
    Option 2: start_all.bat
    Option 3: Manually start backend and frontend in separate terminals
        """)
        return 0
    else:
        print(f"⚠️  {checks_total - checks_passed} checks failed. Please fix issues above.")
        print("""
Quick fixes:
  - Backend dependencies: cd backend && pip install -r requirements.txt
  - Frontend dependencies: cd frontend && npm install
  - Python: Install from https://www.python.org/downloads/
  - Node.js: Install from https://nodejs.org/
        """)
        return 1

if __name__ == '__main__':
    sys.exit(main())
