"""Retrieve public Nuyts benchmark files; verify every SHA-256 before use."""
from pathlib import Path
import csv,json,hashlib,urllib.request
ROOT=Path(__file__).resolve().parents[1]
def fetch(node,file_id,dest,expected):
 if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest()==expected:return
 url=f'https://files.osf.io/v1/resources/{node}/providers/osfstorage/{file_id}/?action=download'
 request=urllib.request.Request(url,headers={'User-Agent':'BIM-subsystem-verification/0.16'})
 with urllib.request.urlopen(request,timeout=60) as response:data=response.read()
 if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('Hash mismatch for '+dest.name)
 dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
for r in json.loads((ROOT/'validation/source_manifest.json').read_text(encoding='utf-8')):
 fetch('j7raf',r['id'],ROOT/r['path'],r['sha256'])
for r in csv.DictReader((ROOT/'validation/ifc/download_manifest.csv').open(encoding='utf-8')):
 dest=ROOT/'validation/ifc'/r['local_path'];fetch('jkh2z',dest.stem,dest,r['sha256'])
print('Verified external source hashes; originals retain their provider licence')
