import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

# Load .env from project root so FINNHUB_API_KEY, OLLAMA_*, etc. are available
load_dotenv(ROOT / ".env")

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
