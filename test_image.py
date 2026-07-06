import base64
import requests

IMAGE_PATH = r"C:\Users\Nishant\Desktop\WhatsApp Image 2026-07-02 at 10.02.43 AM.jpeg"

with open(IMAGE_PATH, "rb") as f:
    image_bytes = f.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

payload = {
    "text": "",
    "image_base64": image_base64
}

response = requests.post("http://127.0.0.1:8000/ask", json=payload)

print("Status Code:", response.status_code)
print("Response:")
print(response.json())