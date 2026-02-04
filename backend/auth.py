import os
from fastapi import HTTPException, status
from dotenv import load_dotenv

# Load env var if not already loaded (though main.py handles it, good practice for standalone testing)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

API_KEY = os.getenv("API_KEY")

def verify_api_key(client_key: str):
    if not client_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key missing"
        )

    if client_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )
