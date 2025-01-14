import sys
import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image

# Dodanie repozytorium YOLOv5 do ścieżki
sys.path.append('./yolov5')
from models.common import DetectMultiBackend  # Kluczowy import

# Wczytaj model YOLOv5
@st.cache_resource
def load_model(model_path):
    try:
        model = DetectMultiBackend(model_path, device='cpu')  # Wymaga repozytorium YOLOv5
        return model
    except Exception as e:
        st.error(f"Błąd podczas ładowania modelu: {e}")
        return None

# Funkcja do detekcji
def detect_characters(image, model):
    results = model(image)  # Bezpośrednie użycie modelu
    img_drawn = np.array(image).copy()

    # Rysowanie wyników
    for *box, conf, cls in results.xyxy[0].cpu().numpy():  # Wyniki detekcji
        label = f'{model.names[int(cls)]} {conf:.2f}'
        cv2.rectangle(img_drawn, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        cv2.putText(img_drawn, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return Image.fromarray(img_drawn)

# Główna logika aplikacji
st.title("Rozpoznawanie Postaci z YOLOv5")

uploaded_file = st.file_uploader("Wybierz zdjęcie", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Przesłane zdjęcie")

    model = load_model('./yolov5/best.pt')
    if model:
        with st.spinner("Rozpoznawanie..."):
            detected_image = detect_characters(image, model)
        st.image(detected_image, caption="Wynik rozpoznawania")
    else:
        st.error("Model nie został załadowany.")
