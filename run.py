import sys
import os

# Add the current directory to sys.path
root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)

import uvicorn
from app.main import app

if __name__ == "__main__":
    # Hugging Face Spaces typically use port 7860
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
