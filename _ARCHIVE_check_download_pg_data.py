import os
import yaml
import powergenome
# os.chdir("/Users/melek/Archive-Switch-USA-PG/Switch-USA-PG-1")

with open("pg_data.yml", "r") as f:
    settings = yaml.safe_load(f)

print("Checking downloaded folders:")
for folder in settings.get("download_folders", {}).keys():
    if os.path.exists(folder):
        print(f" ✅ Found folder: {folder}")
    else:
        print(f" ❌ Missing folder: {folder}")

print("\nChecking downloaded files:")
for file in settings.get("download_files", {}).keys():
    if os.path.exists(file):
        print(f" ✅ Found file: {file}")
    else:
        print(f" ❌ Missing file: {file}")

env_path = os.path.join(powergenome.__path__[0], ".env")
print("\nChecking .env file:")
if os.path.exists(env_path):
    print(f" ✅ Found .env file at: {env_path}")
else:
    print(f" ❌ Missing .env file at: {env_path}")

print("\nChecking case_settings env.yml files:")
for model in settings.get("resource_groups", {}).keys():
    path = os.path.join("MIP_results_comparison", "case_settings", model, "env.yml")
    if os.path.exists(path):
        print(f" ✅ Found env.yml: {path}")
    else:
        print(f" ❌ Missing env.yml: {path}")