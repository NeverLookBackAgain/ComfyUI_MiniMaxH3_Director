"""Instance-local startup configuration for the measured C1 combination."""
import os
from pathlib import Path
import json
def apply():
    config=json.loads((Path(__file__).with_name("h3_c1_startup.json")).read_text(encoding="utf-8"))
    os.environ.pop("PYTORCH_ALLOC_CONF",None)
    for key,value in config["environment"].items():os.environ[key]=value
    # Whitelist the measured acceleration pack for this instance only.
    import sys
    names = config.get("acceleration_plugins", [])
    if "--disable-all-custom-nodes" in sys.argv:
        if "--whitelist-custom-nodes" not in sys.argv:
            sys.argv.extend(["--whitelist-custom-nodes", *names])
        else:
            start = sys.argv.index("--whitelist-custom-nodes") + 1
            end = next((i for i in range(start, len(sys.argv)) if sys.argv[i].startswith("--")), len(sys.argv))
            current = sys.argv[start:end]
            sys.argv[end:end] = [name for name in names if name not in current]
    print("H3 C1 Desktop startup: allocator="+os.environ["PYTORCH_CUDA_ALLOC_CONF"]+" reference_cache="+os.environ["EXPERIMENT_IMAGE_ENCODE_CACHE"],flush=True)
