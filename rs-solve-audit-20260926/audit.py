"""Read-only source audit. Outputs are written beside this file, never into rs-solve."""
from pathlib import Path
from collections import Counter, defaultdict
import ast, csv, hashlib, importlib.util, json, math, sys
from PIL import Image

ROOT = Path(__file__).resolve().parents[1] / 'rs-solve'
OUT = Path(__file__).resolve().parent

def read_csv(name):
    with (ROOT / name).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return reader.fieldnames, rows

def run_audit():
    result = {'scope': str(ROOT)}
    result['inventory'] = [dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(ROOT.rglob('*')) if p.is_file()]
    duplicate_json_keys = []
    def unique_pairs(pairs):
        d = {}
        for k,v in pairs:
            if k in d: duplicate_json_keys.append(k)
            d[k]=v
        return d
    sizes = json.loads((ROOT/'physical_sizes.json').read_text(encoding='utf-8-sig'), object_pairs_hook=unique_pairs)
    questions = json.loads((ROOT/'mau_7_questions.json').read_text(encoding='utf-8-sig'))
    jsonl = [json.loads(s) for s in (ROOT/'mau_7_questions.jsonl').read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    tables = {}
    for name in ['merged_labels.csv','physical_sizes.csv','raw_labels_of_5dataset.csv']:
        fields, rows = read_csv(name)
        keys = [(r['dataset_name'],r['original_class_id']) if 'dataset_name' in r else r['class_id'] for r in rows]
        tables[name] = {'rows':len(rows),'columns':fields,'duplicate_keys':[str(k) for k,v in Counter(keys).items() if v>1],
            'malformed_rows':[i+2 for i,r in enumerate(rows) if None in r or any(v is None for v in r.values())]}
    _, pc = read_csv('physical_sizes.csv'); _, merged = read_csv('merged_labels.csv'); _, raw = read_csv('raw_labels_of_5dataset.csv')
    result['tables'] = tables
    result['raw_dataset_counts'] = dict(Counter(r['dataset_name'] for r in raw))
    result['json'] = {'classes':len(sizes), 'duplicate_keys':duplicate_json_keys,
        'class_id_key_mismatches':[k for k,v in sizes.items() if k!=v['class_id']],
        'inconsistent_record_keys':[k for k,v in sizes.items() if set(v)!=set(next(iter(sizes.values())))],
        'invalid_numeric_intervals':[k for k,v in sizes.items() if not(0<v['L_min_m']<=v['L_typ_m'])]}
    csv_by_id = {r['class_id']:r for r in pc}
    merged_by_id = {r['class_id']:r for r in merged}
    result['keyset_differences'] = {'json_vs_csv':sorted(set(sizes)^set(csv_by_id)), 'json_vs_merged': sorted(set(sizes)^set(merged_by_id))}
    differences = []
    names = {'display_name_en':'canonical_name_en','display_name_vi':'vietnamese_name'}
    for cid, v in sizes.items():
        for field, val in v.items():
            if field=='dataset_mappings':
                for ds, label in val.items():
                    expected = csv_by_id[cid][ds+'_label']; expected = None if expected=='-' else expected
                    if label!=expected: differences.append([cid,ds,label,expected])
            else:
                expected = csv_by_id[cid][names.get(field,field)]
                if field in ['L_min_m','L_typ_m']: expected=float(expected)
                if val!=expected: differences.append([cid,field,val,expected])
    result['csv_json_differences'] = differences
    result['csv_merged_differences'] = [[cid,field,row[field],merged_by_id[cid].get(field)] for cid,row in csv_by_id.items()
        for field in set(row)&set(merged_by_id[cid]) if row[field]!=merged_by_id[cid][field]]
    ds_names = {'dota':'DOTA-v2.0','isaid':'iSAID','dior':'DIOR','visdrone':'VisDrone','xview':'xView'}
    mappings = defaultdict(list); unknown = []; count_errors=[]
    for cid,v in sizes.items():
        for ds,label in v['dataset_mappings'].items():
            if label is None: continue
            for item in label.split(','):
                item = item.strip(); mappings[(ds,item)].append(cid)
                if not any(r['dataset_name']==ds_names[ds] and r['original_class_name']==item for r in raw):
                    unknown.append({'canonical':cid,'dataset':ds,'label':item})
        expected=sum(x is not None for x in v['dataset_mappings'].values())
        if int(merged_by_id[cid]['num_datasets_present'])!=expected: count_errors.append(cid)
    result['mapping_unknown_in_raw'] = unknown
    result['mapping_multiple_canonical'] = [dict(dataset=k[0],label=k[1],canonical=v) for k,v in mappings.items() if len(v)>1]
    result['unmapped_raw_labels'] = [r for r in raw if not any(k[0]==next(d for d,n in ds_names.items() if n==r['dataset_name']) and k[1]==r['original_class_name'] for k in mappings)]
    result['dataset_count_errors'] = count_errors
    result['dimension_type_counts'] = dict(Counter(v['critical_dimension_type'] for v in sizes.values()))
    result['physical_projection'] = [dict(class_id=k, **{f:v[f] for f in ['L_min_m','L_typ_m','critical_dimension_type','critical_dimension_desc','standard_source','reference_url']}) for k,v in sizes.items()]
    images=[]
    for p in sorted((ROOT/'images').glob('*')):
        with Image.open(p) as im:
            images.append(dict(path=str(p.relative_to(ROOT)),width=im.width,height=im.height,format=im.format,exif_fields=len(im.getexif()),metadata_keys=list(im.info)))
    result['images']=images
    image_gsd=defaultdict(set)
    for q in questions: image_gsd[q['image_path']].add(q['gsd_m'])
    result['questions'] = {'count':len(questions),'kind_counts':dict(Counter(q['kind'] for q in questions)), 'jsonl_equal':questions==jsonl,
        'duplicate_ids':[k for k,v in Counter(q['id'] for q in questions).items() if v>1],
        'invalid_class_ids':[q['id'] for q in questions if q['class_id'] not in sizes],
        'answer_not_in_choices':[q['id'] for q in questions if q['answer'] not in q['choices']],
        'missing_images':[q['id'] for q in questions if not (ROOT/q['image_path']).is_file()],
        'image_gsd_values':{k:sorted(v) for k,v in image_gsd.items()},
        'field_coverage':dict(Counter(k for q in questions for k in q)),
        'answer_positions':[{'kind':q['kind'],'index_0based':q['choices'].index(q['answer'])} for q in questions]}
    tree=ast.parse((ROOT/'build_samples.py').read_text(encoding='utf-8-sig'))
    testtree=ast.parse((ROOT/'test_label_parser.py').read_text(encoding='utf-8-sig'))
    def get_map(tree):
        for n in ast.walk(tree):
            if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['DOTA_CLASS_MAP','LABEL_MAP'] for t in n.targets): return ast.literal_eval(n.value)
    result['parser_map_differences']=[{'id':i,'production':get_map(tree).get(i),'test':get_map(testtree).get(i)} for i in set(get_map(tree))|set(get_map(testtree)) if get_map(tree).get(i)!=get_map(testtree).get(i)]
    # Import functions without top-level stdout mutation, __main__, or bytecode writes.
    kept=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef)) or (isinstance(n,ast.Assign) and not any(isinstance(t,ast.Name) and t.id in ['BASE_DIR','LABELS_DIR','PHYSICAL_SIZES_JSON','OUTPUT_JSON','OUTPUT_JSONL'] for t in n.targets))]
    namespace={'__name__':'audit_subject'}
    exec(compile(ast.Module(body=kept,type_ignores=[]),str(ROOT/'build_samples.py'),'exec'),namespace)
    parse=namespace['parse_dota_label']
    objects={p.stem:parse(str(p)) for p in sorted((ROOT/'labels').glob('*.txt'))}
    result['objects']=objects
    result['actual_question_geometry']=[]
    for q in questions:
        objs=objects[Path(q['image_path']).stem]
        chosen=[o for o in objs if o['bbox']==q['target_box_xyxy']]
        if q['kind']=='Q4': chosen=[o for o in objs if o['cell']==q['target_cell'] and o['class_id']==q['class_id']]
        result['actual_question_geometry'].append({'kind':q['kind'],'rho_from_json':q['rho_px'], 'bbox_minor_px':[o['minor_len_px'] for o in chosen], 'aspect_ratios':[round(o['minor_len_px']/o['major_len_px'],4) for o in chosen]})
    specs=sizes
    result['behavior_checks']={}
    check=result['behavior_checks']
    # Controlled input using real basketball boxes: no eligible confusing pair.
    real_parse=namespace['parse_dota_label']
    basketball_only=[o for o in objects['dota_P1470'] if o['class_id']=='basketball_court']
    namespace['parse_dota_label']=lambda *a,**k:basketball_only
    q2=namespace['generate_q2']('unused','unused',.3,specs)
    check['q2_fallback']={'input':'controlled basketball-only subset of real P1470 labels','answer':q2['answer'],'queried_class':q2['class_id'],'annotated_instances':sum(o['class_id']==q2['class_id'] for o in basketball_only)}
    namespace['parse_dota_label']=real_parse
    # Existing P1053 at a lower effective resolution: outside the Q6 selection band.
    q6=namespace['generate_q6']('images/dota_P1053.jpg',str(ROOT/'labels/dota_P1053.txt'),.5,specs)
    check['q6_below_band']={k:q6[k] for k in ['rho_px','p0_U_px','answer','answerable_by_sensor','unanswerable_reason']}
    orig_parse=namespace['parse_dota_label']
    two=[o for o in objects['dota_P1470'] if o['class_id']=='basketball_court'][:2]
    namespace['parse_dota_label']=lambda *a,**k:two
    q5=namespace['generate_q5']('unused','unused',.3,specs)
    check['q5_two_objects']={'answer_box':q5['target_box_xyxy'],'leftmost_box':sorted(two,key=lambda o:o['center'][0])[0]['bbox'],'requested_rank':2}
    namespace['parse_dota_label']=orig_parse
    for name in ['generate_q4','generate_q5']:
        try: namespace[name]('images/dota_P1142.jpg',str(ROOT/'labels/dota_P1142.txt'),.25,specs)
        except Exception as exc: check[name+'_single_object_error']=type(exc).__name__+': '+str(exc)
    return result

if __name__=='__main__':
    result=run_audit()
    (OUT/'audit_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['inventory','physical_projection','objects']},ensure_ascii=False,indent=2))
