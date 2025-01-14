import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image

# Wczytaj model YOLOv5
@st.cache_resource
def load_model(model_path):
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
        return model
    except Exception as e:
        st.error(f"Błąd podczas ładowania modelu: {e}")
        return None

# Funkcja do rozpoznawania postaci
def detect_characters(image, model):
    # Konwertuj obraz do rozmiaru 416x416 pikseli, zachowując proporcje
    img_resized = np.array(image.resize((416, 416)))

    # Wykonaj detekcję
    results = model(img_resized)

    # Przetwórz wyniki
    detections = results.pred[0].cpu().numpy()  # Zapewnij zgodność z CPU
    img_drawn = img_resized.copy()  # Kopia obrazu dla rysowania

    for *box, conf, cls in detections:
        label = f'{model.names[int(cls)]} {conf:.2f}'
        color = (0, 255, 0)  # Zielona ramka i tekst
        cv2.rectangle(img_drawn, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        cv2.putText(img_drawn, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return Image.fromarray(img_drawn)

# Ustawienia aplikacji
st.set_page_config(page_title="Rozpoznawanie Postaci z YOLOv5", layout="centered")
st.markdown("<h1 style='text-align: center; color: white;'>Rozpoznawanie Postaci z YOLOv5</h1>", unsafe_allow_html=True)
st.write("## Prześlij zdjęcie, aby rozpoznać postacie.")
st.markdown("""
    <style>
        body { background-color: #2C2C2C; color: white; }
        .css-1d391kg { text-align: center; }
        .stButton>button { background-color: #4CAF50; color: white; }
    </style>
    """, unsafe_allow_html=True)

# Przesyłanie pliku
uploaded_file = st.file_uploader("Wybierz zdjęcie", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Wyświetl przesłane zdjęcie
    image = Image.open(uploaded_file)
    st.image(image, caption="Przesłane zdjęcie", use_container_width=True)

    # Wczytaj model
    model_path = 'best.pt'


        # Wyświetl wyniki w osobnej sekcji
        st.markdown("## Wynik rozpoznawania")
        st.image(detected_image, caption="Wynik rozpoznawania", use_container_width=True)
    else:
        st.error("Nie można wykonać rozpoznawania, ponieważ model nie został załadowany.")
