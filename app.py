import streamlit as st
import requests
import io
import random
from PIL import Image

st.set_page_config(page_title="Realtime FLUX-RU", page_icon="🎨")

API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
HF_TOKEN = st.secrets["HF_TOKEN"]

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def generate(prompt, w, h, steps, seed=None):
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "width": w, 
            "height": h,
            "num_inference_steps": steps,
            "guidance_scale": 0.0,
            "seed": seed
        }
    }
    
    response = requests.post(API_URL, json=payload, headers=headers, timeout=180)
    response.raise_for_status()
    
    return Image.open(io.BytesIO(response.content)), seed

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

# Настройки seed
with st.expander("⚙️ Дополнительные настройки"):
    use_random_seed = st.checkbox("🎲 Случайный seed (для разнообразия)", value=True)
    if not use_random_seed:
        manual_seed = st.number_input("🌱 Фиксированный seed", 
                                     min_value=0, max_value=2**32-1, 
                                     value=42, step=1)
    else:
        manual_seed = None

if st.button("🚀 Сгенерировать изображение", type="primary"):
    if not prompt.strip():
        st.error("❌ Пожалуйста, введите описание изображения")
    else:
        with st.spinner("🎨 Генерируем изображение..."):
            try:
                seed_to_use = None if use_random_seed else manual_seed
                img, used_seed = generate(prompt, w, h, steps, seed_to_use)
                
                st.image(img, caption="✨ Сгенерированное изображение")
                st.success("✅ Готово! Изображение сгенерировано успешно")
                
                # Информация о генерации
                st.info(f"""
                **Параметры генерации:**
                - 📝 Промпт: {prompt[:50]}{"..." if len(prompt) > 50 else ""}
                - 📐 Размер: {w}×{h}
                - 🔄 Шагов: {steps}
                - 🎲 Seed: {used_seed}
                """)
                
                # Кнопка для повтора с тем же seed
                if st.button(f"🔄 Повторить с seed {used_seed}", key="repeat"):
                    st.rerun()
                
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
        "Космический корабль в далекой галактике",
        "Средневековый замок в тумане",
        "Робот в стиле стимпанк",
        "Подводный город с русалками"
    ]
    
    for example in examples:
        if st.button(f"📋 {example}", key=f"example_{example}"):
            # Заменяем текст в поле промпта
            st.text_area("temp", example, key="temp_prompt", label_visibility="hidden")

st.markdown("---")
st.markdown("**🎨 Создано с помощью FLUX.1-schnell от Black Forest Labs**")
