# at the top of /pages/Base_Model_Trainer.py (only if imports fail)
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))