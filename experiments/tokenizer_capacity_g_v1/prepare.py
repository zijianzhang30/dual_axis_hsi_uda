import difflib
import json
import shutil
from pathlib import Path
from common import OUT,BASE,PROJECT,dump,sha

def main():
 OUT.mkdir(exist_ok=False)
 for p in Path(__file__).parent.iterdir():
  if p.is_file():shutil.copy2(p,OUT/p.name)
 (PROJECT/'results/tokenizer_capacity_g_v1').symlink_to(OUT,target_is_directory=True)
 base=json.loads((BASE/'provenance.json').read_text())
 dump(OUT/'provenance.json',{'baseline':str(BASE),'baseline_provenance_sha256':sha(BASE/'provenance.json'),
  'snapshot_sha256':base['snapshot_sha256'],'dataset_sha256':base['dataset_sha256'],
  'study_sha256':{p.name:sha(p) for p in OUT.glob('*.py')},'protocol_sha256':sha(OUT/'PROTOCOL.md')})
 diff=''.join(difflib.unified_diff((BASE/'scorer.py').read_text().splitlines(True),(OUT/'scorer.py').read_text().splitlines(True),fromfile='frozen_A-F/scorer.py',tofile='independent_G/scorer.py'))
 (OUT/'scorer.diff').write_text(diff)
 dump(OUT/'status.json',{'state':'implementation_audit','existing_Pavia_A_F':'untouched','formal_G_started':False})
 print('Prepared independent G audit')
if __name__=='__main__':main()
