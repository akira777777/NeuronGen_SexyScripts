import json, urllib.request, sys
sys.stdout.reconfigure(encoding='utf-8')
from config import get_profile

c = json.load(urllib.request.urlopen('http://127.0.0.1:7861/config', timeout=20))
# find profile radio + cfg/steps
mode=cfg=steps=None
for x in c.get('components', []):
    props=x.get('props') or {}
    choices=props.get('choices')
    label=str(props.get('label') or '')
    flat=[]
    if choices:
        for ch in choices:
            flat.append(str(ch[0] if isinstance(ch,(list,tuple)) else ch))
    if flat and any('Speed' in f or 'Realism' in f for f in flat):
        mode=(x['id'], props.get('value'), flat, label)
    if 'CFG' in label:
        cfg=(x['id'], props.get('value'), label)
    if 'Steps' in label or 'шаг' in label.lower() or 'Шаг' in label or 'детализац' in label.lower():
        steps=(x['id'], props.get('value'), label)
print('MODE', mode)
print('CFG', cfg)
print('STEPS', steps)

# deps for on_profile / load
for i,d in enumerate(c.get('dependencies') or []):
    print(f"dep[{i}] api={d.get('api_name')} inputs={d.get('inputs')} outputs={d.get('outputs')}")

# Call Gradio API queue for profile change
# Gradio 4: POST /call/on_profile_change or queue with fn_index

def call_api(api_name, data):
    req = urllib.request.Request(
        f'http://127.0.0.1:7861/call/{api_name}',
        data=json.dumps({'data': data}).encode('utf-8'),
        headers={'Content-Type':'application/json'},
        method='POST',
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        event_id = json.load(r)['event_id']
    # SSE result
    with urllib.request.urlopen(f'http://127.0.0.1:7861/call/{api_name}/{event_id}', timeout=60) as r:
        body = r.read().decode('utf-8')
    # parse last data: line
    out=None
    for line in body.splitlines():
        if line.startswith('data: '):
            out = json.loads(line[6:])
    return out

for label in ['⚡ Speed', '✨ Realism / Quality']:
    # try get_profile direct first
    p=get_profile(label)
    print('direct', label, p.get('cfg_scale'), p.get('steps'))

# Find api name - may be on_profile_change
apis = [d.get('api_name') for d in (c.get('dependencies') or [])]
print('apis', apis)

# Try calling if present
for api in apis:
    if api and 'profile' in str(api).lower():
        for label in ['⚡ Speed', '✨ Realism / Quality']:
            try:
                res = call_api(api, [label])
                print('API', api, label, '->', res)
            except Exception as e:
                print('API_FAIL', api, label, e)

# Acceptance summary
speed, realism = get_profile('speed'), get_profile('realism')
checks = []
checks.append(('speed_cfg_not_6', float(speed.get('cfg_scale',0)) != 6.0, speed.get('cfg_scale')))
checks.append(('realism_cfg_6', float(realism.get('cfg_scale',0)) == 6.0, realism.get('cfg_scale')))
checks.append(('realism_steps_28', int(realism.get('steps',0)) == 28, realism.get('steps')))
checks.append(('emoji_speed', float(get_profile('⚡ Speed').get('cfg_scale',0)) == float(speed.get('cfg_scale')), get_profile('⚡ Speed').get('cfg_scale')))
checks.append(('emoji_realism', float(get_profile('✨ Realism / Quality').get('cfg_scale',0)) == 6.0, get_profile('✨ Realism / Quality').get('cfg_scale')))
from prompts.onlyfans_presets import strip_beauty_tokens, build_of_prompt
checks.append(('beauty_strip', 'perfect skin' not in strip_beauty_tokens('x, perfect skin, y').lower(), True))
pos,neg = build_of_prompt()
checks.append(('hand_neg', 'six fingers' in neg, True))
# pytest already 24
print('---ACCEPTANCE---')
fail=0
for name,ok,val in checks:
    print(('PASS' if ok else 'FAIL'), name, val)
    fail += (0 if ok else 1)
print('FAILS', fail)
# note UI default cfg before load fires
if cfg:
    print('NOTE_UI_CFG_INITIAL', cfg[1], '(demo.load should set 6.0 after page open)')
