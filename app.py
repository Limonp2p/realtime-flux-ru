import streamlit as st
import requests
import io
from PIL import Image

st.set_page_config(page_title="Realtime FLUX-RU", page_icon="🎨")

API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
HF_TOKEN = st.secrets["HF_TOKEN"]

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def generate(prompt, w, h, steps):
    payload = {
        "inputs": prompt,
        "parameters": {
            "width": w, 
            "height": h,
            "num_inference_steps": steps,
            "guidance_scale": 0.0
        }
    }
    
    response = requests.post(API_URL, json=payload, headers=headers, timeout=180)
    response.raise_for_status()
    
    # HuggingFace API возвращает изображение в бинарном формате
    return Image.open(io.BytesIO(response.content))

st.title("🎨 Realtime FLUX — RU версия")
st.markdown("Генерация изображений с помощью FLUX.1-schnell")

prompt = st.text_area(
    "📝 Описание изображения на русском:",
    "Красивый закат над морем в стиле импрессионизма",
    height=100
)

col1, col2, col3 = st.columns(3)
w = col1.slider("📏 Ширина", 512, 1024, 768, 64)
h = col2.slider("📐 Высота", 512, 1024, 768, 64) 
steps = col3.slider("🔄 Шаги", 1, 4, 2)

if st.button("🚀 Сгенерировать изображение", type="primary"):
    if not prompt.strip():
        st.error("❌ Пожалуйста, введите описание изображения")
    else:
        with st.spinner("🎨 Генерируем изображение..."):
            try:
                img = generate(prompt, w, h, steps)
                st.image(img, caption="✨ Сгенерированное изображение")
                st.success("✅ Готово! Изображение сгенерировано успешно")
                
                # Информация о генерации
                st.info(f"""
                **Параметры генерации:**
                - 📝 Промпт: {prompt[:50]}{"..." if len(prompt) > 50 else ""}
                - 📐 Размер: {w}×{h}
                - 🔄 Шагов: {steps}
                """)
                
            except requests.exceptions.RequestException as e:
                st.error("❌ Ошибка подключения к API. Попробуйте позже.")
            except Exception as e:
                st.error(f"❌ Произошла ошибка: {str(e)}")

# Примеры промптов
with st.expander("💡 Примеры промптов"):
    examples = [
        "Красивый закат над морем в стиле импрессионизма",
        "Футуристический город с летающими машинами",
        "Портрет кота в костюме астронавта", 
        "Волшебный лес с светящимися грибами",
        "Космический корабль в далекой галактике"
    ]
    for example in examples:
        if st.button(f"📋 {example}", key=example):
            st.rerun()

st.markdown("---")
st.markdown("**🎨 Создано с помощью FLUX.1-schnell от Black Forest Labs**")
