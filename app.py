import subprocess
import sys

subprocess.run([
    sys.executable, "-m", "pip", "install",
    "--upgrade", "pip"
], check=True)

subprocess.run([
    sys.executable, "-m", "pip", "install",
    "-r", "gradio_demo/requirements.txt"
], check=True)

subprocess.run([
    sys.executable, "gradio_demo/download_models.py"
], check=True)

exec(open("gradio_demo/app_multi.py").read())
