"""Rebuild the included exact S kernel with an explicit local compiler."""
from pathlib import Path
import argparse,json,os,shutil,sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'lib'))
from execution import configure,sha

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--compiler',required=True)
    ap.add_argument('--install',action='store_true',help='Replace bundled native binary after a successful build')
    args=ap.parse_args();configure(compiler=args.compiler)
    from route_s_clean.source_order_native import _build
    built=_build()
    report=dict(path=str(built),sha256=sha(built),compiler=args.compiler,
                installed=False,note='Sources/flags unchanged. Cross-platform bitwise equivalence requires revalidation.')
    if args.install:
        destination=ROOT/'native'/('source_order.dll' if os.name=='nt' else 'source_order.so')
        if destination.exists():
            backup=destination.with_name(destination.name+'.'+sha(destination)[:12]+'.previous')
            if not backup.exists():shutil.copyfile(destination,backup)
        shutil.copyfile(built,destination);report.update(installed=True,path=str(destination))
    print(json.dumps(report),flush=True)
if __name__=='__main__':main()
