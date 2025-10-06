import streamlit as st
import requests, base64, io
from PIL import Image

st.set_page_config(page_title="Realtime FLUX-RU", page_icon="🎨")

API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
HF_TOKEN = st.secrets["HF_TOKEN"]      # токен добавится в Secrets

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def generate(prompt, w, h, steps):
    payload = {
        "inputs": prompt,
        "parameters": {"width": w, "height": h,
                       "num_inference_steps": steps,
                       "guidance_scale": 0.0}
    }
    r = requests.post(API_URL, json=payload, headers=headers, timeout=180)
    r.raise_for_status()
    img_b64 = r.json()[0]["image_base64"]
    return Image.open(io.BytesIO(base64.b64decode(img_b64)))

st.title("🎨 Realtime FLUX — RU версия")
prompt = st.text_area("Описание на русском",
                      "Красивый закат над морем в стиле импрессионизма")
c1, c2, c3 = st.columns(3)
w = c1.slider("Ширина", 512, 1024, 768, 64)
h = c2.slider("Высота", 512, 1024, 768, 64)
steps = c3.slider("Шаги", 1, 4, 2)

if st.button("🚀 Сгенерировать"):
    with st.spinner("Генерируем…"):
        img = generate(prompt, w, h, steps)
        st.image(img, caption="Результат")
        st.success("Готово!")
