import re
from pathlib import Path

def get_pnum_from_snp(path: str | Path):
    return int(re.match(r".[sS](\d+)[pP]", Path(path).suffix).group(1))