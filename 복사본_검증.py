"""Check snapshot integrity, local imports, full pipeline and illustrative matrices.

This is packaging verification, not a performance comparison simulator.
"""
from pathlib import Path
import hashlib
import importlib
import json
import re
import xml.etree.ElementTree as ET
from 실행 import ROOT,load_snapshot


def main():
    manifest=load_snapshot()
    imports=[]
    for entry in manifest['files']:
        path=ROOT/entry['copy']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256'],path
        module=importlib.import_module(entry['module'])
        assert Path(module.__file__).resolve()==path.resolve(),module.__file__
        imports.append(entry['module'])
    from map_data import GridMap,Node,Edge,write_json
    from whole_map_matrix import encode_whole_map,decode_whole_map
    from spectral_pipeline import solve_spectral_map
    from spectral_target import SpectralTarget
    from matrix_recovery import family_matrix,recover_matrix
    from lc_model import LCModel,decode_circuit
    original=json.loads((ROOT/'04_실행예제/통합예제/전체실행.json').read_text(encoding='utf8'))
    whole=json.loads((ROOT/'04_실행예제/통합예제/지도행렬.json').read_text(encoding='utf8'))
    actual=solve_spectral_map(whole)
    assert actual['complete']
    keys=lambda r:{tuple(v['characteristic_polynomial_coefficients']) for v in r['generation']['spectra']}
    assert keys(actual)==keys(original)
    assert actual['optimal_route_counts']==original['optimal_route_counts']
    grid=decode_whole_map(whole)
    for entry in actual['inverse_results']:
        for solution in entry['result']['solutions']:
            assert decode_circuit(solution['circuit'])==(grid,tuple(solution['counts']))
    write_json(ROOT/'04_실행예제/복사본실행.json',actual)
    tiny=GridMap('matrix-explanation-3-nodes',
                 (Node(0,0,False,0),Node(1,0,True,1),Node(2,0,True,2)),
                 (Edge(0,1,2),Edge(1,2,3)))
    m=(2,2)
    R=family_matrix(LCModel(tiny),m)
    target=SpectralTarget(tuple(R.charpoly().all_coeffs()))
    recovered=recover_matrix(tiny,R,target,max_count=2)
    assert recovered['time_ticks']==13
    assert sum(R.diagonal())==16
    sample=ROOT/'04_실행예제/행렬설명용_3노드'
    write_json(sample/'지도.json',tiny.to_dict())
    write_json(sample/'지도행렬.json',encode_whole_map(tiny))
    write_json(sample/'경로와_모든행렬.json',recovered)
    write_json(sample/'LC회로.json',recovered['circuit'])
    write_json(sample/'목표스펙트럼.json',target.to_dict())
    diagrams=list(ROOT.rglob('흐름도.svg'))
    for path in diagrams:
        svg=ET.parse(path).getroot()
        assert len(svg.findall('{http://www.w3.org/2000/svg}rect'))>=2
    checked_links=0
    for path in ROOT.rglob('*.html'):
        for ref in re.findall(r'(?:href|src)="([^"]+)"',path.read_text(encoding='utf8')):
            if not ref.startswith(('http:','https:','#','data:')):
                assert (path.parent/ref).exists(),(path,ref)
                checked_links+=1
    result={'snapshot_modules_hash_verified':len(imports),'all_imported_from_copy':True,
            'pipeline_complete':actual['complete'],'spectra_exact_match':len(keys(actual)),
            'optimal_route_counts_exact_match':len(actual['optimal_route_counts']),
            'circuit_roundtrip_verified':True,'illustrative_time_ticks':13,
            'svg_diagrams_xml_checked':len(diagrams),'local_html_links_checked':checked_links,
            'is_performance_benchmark':False}
    write_json(ROOT/'05_검증과출처/복사본_검증결과.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
