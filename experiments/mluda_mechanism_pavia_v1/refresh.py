"""Freeze corrected study scripts before any formal run; never touch native snapshot."""
import json
import shutil
from pathlib import Path
from common import OUT, dump, sha

assert not (OUT/'launcher_status.json').exists()
assert not list(OUT.glob('M[0-3]_*'))
for p in Path(__file__).parent.iterdir():
    if p.is_file():
        shutil.copy2(p,OUT/p.name)
manifest=json.loads((OUT/'manifest.json').read_text())
manifest['study_sha256']={p.name:sha(p) for p in OUT.glob('*.py')}
manifest['protocol_sha256']=sha(OUT/'PROTOCOL.md')
dump(OUT/'manifest.json',manifest)
print('Frozen study scripts before formal training')
