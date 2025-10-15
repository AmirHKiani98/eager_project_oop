import subprocess

models = ["pq", "sq", "pw", "ctm", "ltm"]
base_command = [
    "python", "-m", "src.main",
    "--fp-date=20181030",
    "--fp-time=0800_0830",
    "--calibration",
    "--cache-dir=.cache_dt5s_case1",
    "--dt=5"
]

for model in models:
    cmd = base_command + [f'--model={model}']
    with open(f"output_5s_{model}.log", "a") as outfile:
        subprocess.run(cmd, stdout=outfile, stderr=subprocess.STDOUT)