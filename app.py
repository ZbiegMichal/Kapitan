import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image

# Wczytaj model YOLOv5
@st.cache_resource
def load_model(model_path):
    try:
        # Ładowanie modelu lokalnie z pliku .pt
        model = torch.load(model_path, map_location=torch.device('cpu'))
        model.eval()  # Ustawienie modelu w tryb oceny
        return model
    except FileNotFoundError:
        st.error("Nie znaleziono pliku modelu. Upewnij się, że plik 'best.pt' znajduje się w katalogu aplikacji.")
        return None
    except Exception as e:
        st.error(f"Błąd podczas ładowania modelu: {e}")
        return None

# Funkcja do rozpoznawania postaci
def detect_characters(image, model):
    # Konwertuj obraz na format akceptowany przez model
    img_resized = np.array(image.resize((640, 640)))  # Dopasowanie do wymaganego rozmiaru
    img = img_resized / 255.0  # Normalizacja
    img = np.transpose(img, (2, 0, 1))  # Przekształcenie na format CHW
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # Dodanie wymiaru batch
    
    # Wykonaj detekcję
    with torch.no_grad():
        results = model(img)
    
    # Przetwórz wyniki
    detections = results[0]  # Wyniki detekcji
    img_drawn = np.array(image).copy()  # Kopia obrazu dla rysowania
    
    for det in detections:
        box = det[:4].cpu().numpy()  # Współrzędne ramki
        conf = det[4].item()  # Pewność predykcji
        cls = int(det[5].item())  # Klasa
        
        label = f'{model.names[cls]} {conf:.2f}'
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
    model = load_model(model_path)

    if model:
        # Rozpoznawanie postaci
        with st.spinner("Rozpoznawanie postaci..."):
            detected_image = detect_characters(image, model)
        
        # Wyświetl wyniki w osobnej sekcji
        st.markdown("## Wynik rozpoznawania")
        st.image(detected_image, caption="Wynik rozpoznawania", use_container_width=True)
    else:
        st.error("Nie można wykonać rozpoznawania, ponieważ model nie został załadowany.")
