from google.colab import userdata
from pyngrok import ngrok
import subprocess
import time
import os

os.environ['SBER_API_KEY'] = userdata.get('sber_api_key')
NGROK_AUTH_TOKEN = userdata.get('ngrok_token')
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

subprocess.Popen(["streamlit", "run", "app.py", "--server.port", "8501", "--server.headless", "true"])

time.sleep(5)
public_url = ngrok.connect(8501, "http")
print(public_url)
