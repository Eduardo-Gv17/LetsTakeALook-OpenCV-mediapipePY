import sys

import cv2
import mediapipe as mp


# --- 1. Inicialización ---
print("Cargando modelos y assets...")

# Carga la solución de Face Mesh de MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,  # ¡CLAVE para pupilas!
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Inicializa la captura de video (cámara web)
cap = cv2.VideoCapture(0)

# --- 2. Cargar y Cortar la Imagen de "Ojitos" ---

# Carga la imagen CON su canal de transparencia (alfa)
# Si da error aquí, asegúrate de que "ojitos.png" esté en la misma carpeta.
try:
    ojitos_img_raw = cv2.imread("ojitos.png", cv2.IMREAD_UNCHANGED)
    if ojitos_img_raw is None:
        raise IOError
except IOError:
    print("Error: No se pudo cargar la imagen 'ojitos.png'.")
    print("Asegúrate de que esté en la misma carpeta que el script.")
    cap.release()
    sys.exit()

# Obtenemos las dimensiones de la imagen
h_img, w_img, _ = ojitos_img_raw.shape
# La cortamos por la mitad para tener cada ojo por separado
mitad_ancho = w_img // 2
ojo_izq_img = ojitos_img_raw[:, :mitad_ancho]
ojo_der_img = ojitos_img_raw[:, mitad_ancho:]

print("¡Todo cargado! Iniciando cámara. Presiona 'ESC' para salir.")


# --- 3. Función Mágica para Superponer Transparencias ---

def overlay_transparent(background_img, foreground_img, x, y):
    """
    Superpone una imagen (foreground) con transparencia (canal alfa)
    sobre otra imagen (background) en la posición (x, y).
    """
    # Dimensiones de la imagen de fondo
    h_bg, w_bg, _ = background_img.shape

    # Dimensiones de la imagen de primer plano (los ojitos)
    h_fg, w_fg, channels_fg = foreground_img.shape

    # Si la imagen de primer plano no tiene 4 canales (RGBA), no hacemos nada
    if channels_fg != 4:
        return background_img

    # --- Calcular dónde pegar la imagen ---

    # El (x, y) que recibimos es el CENTRO,
    # pero para pegar necesitamos la esquina superior izquierda.
    top = y - h_fg // 2
    left = x - w_fg // 2

    # Límites para no salirnos de la pantalla
    bottom = top + h_fg
    right = left + w_fg

    # Recortar si se sale por arriba/izquierda
    if top < 0:
        h_fg = h_fg + top
        top = 0
    if left < 0:
        w_fg = w_fg + left
        left = 0

    # Recortar si se sale por abajo/derecha
    if bottom > h_bg:
        h_fg = h_bg - top
    if right > w_bg:
        w_fg = w_bg - left

    # Si la imagen está completamente fuera, no hacer nada
    if h_fg <= 0 or w_fg <= 0:
        return background_img

    # Recortar la imagen de primer plano según los nuevos límites
    # (esto es por si te acercas mucho al borde de la cámara)
    foreground_img_clipped = foreground_img[0:h_fg, 0:w_fg]

    # --- El Proceso de Alpha Blending ---

    # 1. Separar la máscara alfa del color
    # La máscara alfa nos dice qué píxeles son transparentes (0) y cuáles opacos (255)
    alpha_mask = foreground_img_clipped[:, :, 3] / 255.0
    # La máscara inversa nos dice qué parte del fondo debe mantenerse
    alpha_mask_inv = 1.0 - alpha_mask

    # 2. Obtener la región de interés (ROI) del fondo donde pegaremos el ojo
    roi = background_img[top:top + h_fg, left:left + w_fg]

    # 3. Quitar el color del emoji de la máscara alfa
    rgb_foreground = foreground_img_clipped[:, :, :3]

    # 4. Combinar todo
    for c in range(0, 3):  # Para cada canal (R, G, B)
        # Parte 1: Multiplicar la máscara por el fondo (lo "borra")
        bg_part = (alpha_mask_inv * roi[:, :, c])
        # Parte 2: Multiplicar la máscara por el primer plano (lo "trae")
        fg_part = (alpha_mask * rgb_foreground[:, :, c])
        # Sumar ambas partes
        roi[:, :, c] = bg_part + fg_part

    # 5. Modificar la imagen de fondo original
    background_img[top:top + h_fg, left:left + w_fg] = roi

    return background_img


# --- 4. Bucle Principal ---
while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1)

    # Guardamos las dimensiones para los cálculos
    h_cam, w_cam, _ = image.shape

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    # Creamos una copia de la imagen para no modificar la original
    # mientras la función de 'overlay' trabaja
    image_con_ojitos = image.copy()

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark

        # --- 5. Calcular Coordenadas y Tamaño para CADA OJO ---

        # Puntos de referencia para el ojo IZQUIERDO del usuario
        # (Es el ojo de la derecha en la pantalla)
        # 473: Centro de la pupila izquierda
        # 133: Esquina interior del ojo izquierdo
        # 33: Esquina exterior del ojo izquierdo

        pupila_izq_x = int(landmarks[473].x * w_cam)
        pupila_izq_y = int(landmarks[473].y * h_cam)

        # Calculamos el ancho del ojo usando la distancia entre la esquina exterior e interior
        # Nos aseguramos de que el resultado sea siempre positivo con abs()
        ancho_ojo_izq_real = int(abs(landmarks[33].x * w_cam - landmarks[133].x * w_cam))

        # Ajustamos el tamaño del emoji.
        # Multiplicamos por un factor (ej. 3.0) para que sea más grande. ¡Juega con este número!
        factor_escala = 2
        ancho_emoji_izq = int(ancho_ojo_izq_real * factor_escala)

        # Redimensionamos la imagen del ojo emoji (manteniendo proporción)
        h_ojo_izq, w_ojo_izq, _ = ojo_izq_img.shape
        if w_ojo_izq > 0:  # Evitar división por cero
            ratio_izq = ancho_emoji_izq / w_ojo_izq
            alto_emoji_izq = int(h_ojo_izq * ratio_izq)
        else:
            ratio_izq = 0
            alto_emoji_izq = 0

        # Aseguramos que las dimensiones sean válidas antes de redimensionar y pegar
        if ancho_emoji_izq > 0 and alto_emoji_izq > 0:
            ojo_izq_redimensionado = cv2.resize(ojo_izq_img, (ancho_emoji_izq, alto_emoji_izq))
            # Pegamos la imagen
            image_con_ojitos = overlay_transparent(image_con_ojitos, ojo_izq_redimensionado, pupila_izq_x, pupila_izq_y)

        # Puntos de referencia para el ojo DERECHO del usuario
        # (Es el ojo de la izquierda en la pantalla)
        # 468: Centro de la pupila derecha
        # 263: Esquina interior del ojo derecho
        # 362: Esquina exterior del ojo derecho

        pupila_der_x = int(landmarks[468].x * w_cam)
        pupila_der_y = int(landmarks[468].y * h_cam)

        # Calculamos el ancho del ojo derecho de la misma manera
        ancho_ojo_der_real = int(abs(landmarks[362].x * w_cam - landmarks[263].x * w_cam))

        ancho_emoji_der = int(ancho_ojo_der_real * factor_escala)  # Usamos el mismo factor de escala

        # Redimensionamos
        h_ojo_der, w_ojo_der, _ = ojo_der_img.shape
        if w_ojo_der > 0:  # Evitar división por cero
            ratio_der = ancho_emoji_der / w_ojo_der
            alto_emoji_der = int(h_ojo_der * ratio_der)
        else:
            ratio_der = 0
            alto_emoji_der = 0

        # Aseguramos que las dimensiones sean válidas antes de redimensionar y pegar
        if ancho_emoji_der > 0 and alto_emoji_der > 0:
            ojo_der_redimensionado = cv2.resize(ojo_der_img, (ancho_emoji_der, alto_emoji_der))
            # Pegamos la imagen
            image_con_ojitos = overlay_transparent(image_con_ojitos, ojo_der_redimensionado, pupila_der_x, pupila_der_y)

    # --- 6. Mostrar la Imagen ---
    cv2.imshow('Filtro de Ojitos Wholesome - por MediaPipe', image_con_ojitos)

    if cv2.waitKey(5) & 0xFF == 27:  # Tecla ESC
        break

# --- 7. Limpieza ---
cap.release()
cv2.destroyAllWindows()