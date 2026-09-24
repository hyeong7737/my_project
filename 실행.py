"""Standalone entry point for the verbatim copied production modules."""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parent


def load_snapshot():
    manifest=json.loads((ROOT/'05_검증과출처'/'코드_복사목록.json').read_text(encoding='utf8'))
    for entry in manifest['files']:
        folder=str((ROOT/entry['copy']).parent)
        if folder not in sys.path: sys.path.insert(0,folder)
    return manifest


def main():
    load_snapshot()
    from map_data import GridMap,ValidationError,write_json
    from whole_map_matrix import encode_whole_map
    from spectral_pipeline import solve_spectral_map
    parser=argparse.ArgumentParser(description='전체 지도 → 스펙트럼 → 최적 회로/경로 (복사본 독립 실행)')
    parser.add_argument('--input',type=Path,default=ROOT/'04_실행예제/통합예제/지도행렬.json')
    parser.add_argument('--output',type=Path,default=ROOT/'04_실행예제/복사본실행.json')
    parser.add_argument('--roots',action='store_true')
    for option,default in [('max-forests',10000),('max-spectra',1000),('timeout-ms',10000),
                           ('total-timeout-ms',60000),('inverse-max-states',200000),('inverse-max-solutions',10000)]:
        parser.add_argument('--'+option,type=int,default=default)
    args=parser.parse_args()
    try:
        data=json.loads(args.input.read_text(encoding='utf-8-sig'))
        if data.get('schema')=='lc-grid-v1': data=encode_whole_map(GridMap.from_dict(data))
        result=solve_spectral_map(data,max_forests=args.max_forests,max_spectra=args.max_spectra,
                                 timeout_ms=args.timeout_ms,total_timeout_ms=args.total_timeout_ms,
                                 inverse_max_states=args.inverse_max_states,
                                 inverse_max_solutions=args.inverse_max_solutions,include_roots=args.roots)
        write_json(args.output,result)
        print('complete=',result['complete'],'spectra=',len(result['generation']['spectra']),
              'optimal_routes=',len(result['optimal_route_counts']),'output=',args.output)
        return 0 if result['complete'] else 3
    except (ValidationError,OSError,ValueError) as error:
        print(str(error),file=sys.stderr)
        return 2


if __name__=='__main__':raise SystemExit(main())
