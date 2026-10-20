"""Export paper metrics from newly generated runs without changing their outputs."""
import csv
import json
from pathlib import Path
from tutus.evaluation.summary import harmonic_mean


def report(root):
    from tutus.cli import PAPER, MODELS
    rows=[]
    for path in sorted(root.rglob('*metrics.json')):
        relative=path.relative_to(root)
        ident=next((part for part in relative.parts if any(part.startswith(model+'-') for model in MODELS)),None)
        if ident is None:continue
        model=next(model for model in sorted(MODELS,key=len,reverse=True) if ident.startswith(model+'-'))
        remainder=ident[len(model)+1:].split('-')
        method=remainder[0];objective=remainder[1] if len(remainder)>1 else ''
        filename=path.name.split('___');checkpoint=filename[0] if len(filename)>1 else 'training-end'
        split=filename[1] if len(filename)>2 else 'TUTUS-DEV' if 'post-training' in relative.parts else 'TUTUS-TS-P'
        values=json.loads(path.read_text())
        def scalar(key):
            value=values.get(key)
            return value[-1] if isinstance(value,list) and value else value
        for group,prefix in [('all','average'),('seen','base'),('unseen','novel')]:
            rss,cdp,jdr=(scalar(f'{prefix}_{name}') for name in ['asr','cdp','jdr'])
            if any(value is None for value in [rss,cdp,jdr]):continue
            rows.append({'model':model,'method':method,'objective':objective,'checkpoint':checkpoint,'split':split,'group':group,'rss':rss,'cdp':cdp,'jdr':jdr,'hm':harmonic_mean(rss,cdp,jdr),'irm_clipped_product':1-rss*jdr,'metrics_file':relative.as_posix()})
    folder=root/'report';folder.mkdir(parents=True,exist_ok=True)
    output=folder/'metrics.csv'
    fields=['model','method','objective','checkpoint','split','group','rss','cdp','jdr','hm','irm_clipped_product','metrics_file']
    with output.open('w',newline='') as f:
        writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(rows)
    gaps=[]
    for row in rows:
        if row['group']!='seen':continue
        unseen=next((r for r in rows if all(r[k]==row[k] for k in ['model','method','objective','checkpoint','split']) and r['group']=='unseen'),None)
        if unseen:gaps.append({**{k:row[k] for k in ['model','method','objective','checkpoint','split']},'seen_hm':row['hm'],'unseen_hm':unseen['hm'],'gap_percentage_points':100*(row['hm']-unseen['hm'])})
    with (folder/'seen_unseen.csv').open('w',newline='') as f:
        fields=['model','method','objective','checkpoint','split','seen_hm','unseen_hm','gap_percentage_points'];writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(gaps)
    utility=[]
    for path in sorted((root/'utility').rglob('*_results.json')) if (root/'utility').is_dir() else []:
        document=json.loads(path.read_text())
        for task,results in document.get('results',{}).items():
            for metric,value in results.items():
                if isinstance(value,(float,int)):utility.append({'run':path.relative_to(root/'utility').parts[0],'task':task,'metric':metric,'value':value})
    with (folder/'utility.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,['run','task','metric','value']);writer.writeheader();writer.writerows(utility)
    attack=[]
    for path in sorted(root.rglob('rebuttal_summary.csv')):
        with path.open(newline='') as f:
            for row in csv.DictReader(f):attack.append({'source':path.relative_to(root).as_posix(),**row})
    if attack:
        fields=list(dict.fromkeys(k for row in attack for k in row))
        with (folder/'attacks_and_benign.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(attack)
    print(f'Exported {len(rows)} metric rows, {len(gaps)} seen/unseen comparisons, {len(utility)} utility scores, and {len(attack)} stress-test rows to {folder}')
    print('HM is computed from RSS/CDP/JDR using the standard harmonic mean. Stored metrics and manuscript numbers are not modified.')
