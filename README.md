# 👀 Filtro de Ojitos Wholesome (Let's Take a Look)

Este proyecto es un script de Python que aplica el popular filtro de "ojitos wholesome" (el emoji 👀) a tu rostro en tiempo real, utilizando tu cámara web. El script detecta tus pupilas y superpone el emoji, escalándolo dinámicamente.

Está basado en el meme "Let's take a look" / "ojitos void".


## 🚀 Características

* Detección facial en tiempo real.
* Seguimiento preciso de pupilas (landmarks 468 y 473) usando MediaPipe.
* Superposición de la imagen `ojitos.png` con canal alfa (transparencia).
* Escalado dinámico del emoji basado en el tamaño de tus ojos (distancia entre los bordes del ojo).

---

## 🛠️ Tecnologías Usadas

* **Python 3** (Desarrollado con 3.11)
* **OpenCV (cv2):** Para capturar el video de la cámara web, manejar las imágenes y dibujar en pantalla.
* **MediaPipe:** Para la detección facial y el seguimiento de los 478 puntos de referencia (landmarks) de la cara.

---

## 📦 Instalación y Ejecución

Sigue estos pasos para ejecutar el proyecto en tu máquina local.

### 1. Prerrequisitos

* Python (3.8 - 3.11 recomendado)
* Una cámara web
* El archivo `ojitos.png` (¡asegúrate de que esté en la misma carpeta!)

### 2. Clona el Repositorio

```bash
git clone https://github.com/Eduardo-Gv17/LetsTakeALook-OpenCV-mediapipePY.git

```
### 3.Instala las Dependencias
Este proyecto requiere opencv-python y mediapipe.
```
pip install opencv-python mediapipe
```

### 4. Ejecuta el script y disfruta!
Mírate a la cámara y ¡listo!

Para salir, presiona la tecla ESC.



