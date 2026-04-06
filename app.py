import os
import subprocess
import sys

if not os.path.exists("./models/antelopev2"):
    subprocess.run([sys.executable, "gradio_demo/download_models.py"], check=True)

if not os.path.exists("./checkpoints/ip-adapter.bin"):
    subprocess.run([sys.executable, "gradio_demo/download_models.py"], check=True)

exec(open("gradio_demo/app_multi.py").read())
