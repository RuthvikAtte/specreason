import importlib

modules = [
    "numpy",
    "openai",
    "datasets",
    "tqdm",
    "transformers",
]

missing = []
for m in modules:
    try:
        importlib.import_module(m)
    except Exception as e:
        missing.append((m, str(e)))

if missing:
    print("MISSING_DEPENDENCIES")
    for name, err in missing:
        print(f"- {name}: {err}")
    raise SystemExit(1)

print("SETUP_SMOKE_TEST_OK")
