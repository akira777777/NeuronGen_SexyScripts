import json, urllib.request, sys, urllib.error
sys.stdout.reconfigure(encoding='utf-8')
c = json.load(urllib.request.urlopen("http://127.0.0.1:7861/config", timeout=15))
for x in c.get("components", []):
    props = x.get("props") or {}
    choices = props.get("choices")
    if not choices:
        continue
    flat = []
    for ch in choices:
        if isinstance(ch, (list, tuple)):
            flat.append(str(ch[0]))
        else:
            flat.append(str(ch))
    joined = " | ".join(flat)
    if any(k in joined for k in ("Speed", "Realism", "Quality")):
        print("MODE", x.get("id"), x.get("type"), "label=", props.get("label"), "value=", props.get("value"))
        print("  choices=", flat)

# dependencies: find which fn updates cfg when mode changes
for i, d in enumerate(c.get("dependencies") or []):
    targets = d.get("targets")
    triggers = d.get("trigger")
    inputs = d.get("inputs")
    outputs = d.get("outputs")
    api = d.get("api_name")
    print(f"dep[{i}] trigger={triggers} api={api} inputs={inputs} outputs={outputs}")

# Try calling queue API: typically /run/predict or /call/{api_name}
# Inspect main.py wiring line ~830
