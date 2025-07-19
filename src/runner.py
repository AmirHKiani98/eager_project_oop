import subprocess

models = ["pq", "sq", "pw"]
base_command = [
    "python", "-m", "src.main",
    "--calibration",
    "--cache-dir=.cache_dt5s",
    "--dt=5"
]

for model in models:
    cmd = base_command + [f'--model={model}']
    with open(f"output_5s_{model}.log", "a") as outfile:
        subprocess.run(cmd, stdout=outfile, stderr=subprocess.STDOUT)