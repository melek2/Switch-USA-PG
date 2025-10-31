import os

env_path = "/Users/melek/Switch-USA-PG/PowerGenome/powergenome/.env"

print(f"Checking paths listed in: {env_path}\n")

with open(env_path, "r") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            var, path = line.split("=", 1)
            path = path.strip("'\"")  # remove quotes
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                print(f"✅ {var} exists: {abs_path}")
            else:
                print(f"❌ {var} MISSING: {abs_path}")
        except ValueError:
            print(f"⚠️ Skipping malformed line: {line}")