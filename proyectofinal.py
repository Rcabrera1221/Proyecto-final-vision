import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime

# Inicializar mediapipe pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Funciones auxiliares para sentadillas
def encontrar_angulo(a, b, c, minVis=0.8):
    if a.visibility > minVis and b.visibility > minVis and c.visibility > minVis:
        bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
        ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
        angulo = np.arccos((np.dot(ba, bc)) / (np.linalg.norm(ba) * np.linalg.norm(bc))) * (180 / np.pi)
        if angulo > 180:
            return 360 - angulo
        else:
            return angulo
    else:
        return -1

def estado_pierna(angulo):
    if angulo < 0:
        return 0  
    elif angulo < 105:
        return 1  
    elif angulo < 150:
        return 2  
    else:
        return 3  

def verificar_rodillas(lm_arr):
    rodilla_izquierda = lm_arr[25]
    rodilla_derecha = lm_arr[26]
    tobillo_izquierdo = lm_arr[27]
    tobillo_derecho = lm_arr[28]

    distancia_rodillas = np.linalg.norm(np.array([rodilla_izquierda.x - rodilla_derecha.x, rodilla_izquierda.y - rodilla_derecha.y]))
    distancia_tobillos = np.linalg.norm(np.array([tobillo_izquierdo.x - tobillo_derecho.x, tobillo_izquierdo.y - tobillo_derecho.y]))

    if distancia_rodillas >= distancia_tobillos * 0.8:
        return True, "Rodillas bien alineadas"
    return False, "Rodillas no alineadas"

def verificar_profundidad(lm_arr):
    cadera = lm_arr[24]
    rodilla = lm_arr[26]

    if cadera.y >= rodilla.y - 0.05: 
        return True, "Buena profundidad"
    return False, "Profundidad insuficiente"

def verificar_espalda(lm_arr):
    hombro_izquierdo = lm_arr[11]
    hombro_derecho = lm_arr[12]
    cadera_izquierda = lm_arr[23]
    cadera_derecha = lm_arr[24]

    angulo_hombro_cadera_izquierdo = encontrar_angulo(hombro_izquierdo, cadera_izquierda, cadera_derecha)
    angulo_hombro_cadera_derecho = encontrar_angulo(hombro_derecho, cadera_derecha, cadera_izquierda)

    if angulo_hombro_cadera_izquierdo >= 60 and angulo_hombro_cadera_derecho >= 60:
        return True, "Alineacion de la espalda correcta"
    return False, "Espalda no recta"

def verificar_alineacion_caderas(lm_arr):
    cadera_izquierda = lm_arr[23]
    cadera_derecha = lm_arr[24]
    rodilla_izquierda = lm_arr[25]
    rodilla_derecha = lm_arr[26]
    tobillo_izquierdo = lm_arr[27]
    tobillo_derecho = lm_arr[28]

    cadera_rodilla_izquierda = np.linalg.norm(np.array([cadera_izquierda.x - rodilla_izquierda.x, cadera_izquierda.y - rodilla_izquierda.y]))
    cadera_tobillo_izquierda = np.linalg.norm(np.array([cadera_izquierda.x - tobillo_izquierdo.x, cadera_izquierda.y - tobillo_izquierdo.y]))
    cadera_rodilla_derecha = np.linalg.norm(np.array([cadera_derecha.x - rodilla_derecha.x, cadera_derecha.y - rodilla_derecha.y]))
    cadera_tobillo_derecha = np.linalg.norm(np.array([cadera_derecha.x - tobillo_derecho.x, cadera_derecha.y - tobillo_derecho.y]))

    if abs(cadera_rodilla_izquierda - cadera_tobillo_izquierda) < 0.2 and abs(cadera_rodilla_derecha - cadera_tobillo_derecha) < 0.2:
        return True, "Caderas bien alineadas"
    return False, "Caderas no alineadas"

def verificar_estabilidad_postura(lm_arr):
    pie_izquierdo = lm_arr[31]
    pie_derecho = lm_arr[32]

    if pie_izquierdo.visibility > 0.5 and pie_derecho.visibility > 0.5:
        return True, "Postura estable al final"
    return False, "Postura inestable al final"

def dibujar_barra_progreso(frame, progreso, x, y, ancho, alto):
    cv2.rectangle(frame, (x, y), (x + ancho, y + alto), (128, 128, 128), -1)
    cv2.rectangle(frame, (x, y + alto - int(alto * progreso)), (x + ancho, y + alto), (255, 0, 0), -1)
    cv2.putText(frame, f"{int(progreso * 100)}%", (x + 5, y + alto - int(alto * progreso) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def calcular_precision_sentadilla(lm_arr, fase):
    puntuacion_total = 0
    puntuacion_maxima = 3
    advertencias = []
    retroalimentacion_positiva = []

    if fase == 'abajo':
        angulo = encontrar_angulo(lm_arr[24], lm_arr[26], lm_arr[28])
        if 60 <= angulo <= 150:
            puntuacion_total += 1
            retroalimentacion_positiva.append("Angulo de rodilla correcto")
        else:
            advertencias.append("Angulo de rodilla incorrecto")

        rodillas_alineadas, mensaje_rodillas = verificar_rodillas(lm_arr)
        if rodillas_alineadas:
            puntuacion_total += 1
            retroalimentacion_positiva.append(mensaje_rodillas)
        else:
            advertencias.append(mensaje_rodillas)

        profundidad_correcta, mensaje_profundidad = verificar_profundidad(lm_arr)
        if profundidad_correcta:
            puntuacion_total += 1
            retroalimentacion_positiva.append(mensaje_profundidad)
        else:
            advertencias.append(mensaje_profundidad)

    elif fase == 'arriba':
        espalda_recta, mensaje_espalda = verificar_espalda(lm_arr)
        if espalda_recta:
            puntuacion_total += 1
            retroalimentacion_positiva.append(mensaje_espalda)
        else:
            advertencias.append(mensaje_espalda)

        caderas_alineadas, mensaje_caderas = verificar_alineacion_caderas(lm_arr)
        if caderas_alineadas:
            puntuacion_total += 1
            retroalimentacion_positiva.append(mensaje_caderas)
        else:
            advertencias.append(mensaje_caderas)

        postura_estable, mensaje_postura = verificar_estabilidad_postura(lm_arr)
        if postura_estable:
            puntuacion_total += 1
            retroalimentacion_positiva.append(mensaje_postura)
        else:
            advertencias.append(mensaje_postura)

    precision = (puntuacion_total / puntuacion_maxima) * 100  # Calcular precisión en porcentaje
    return precision, advertencias, retroalimentacion_positiva

def dibujar_reloj(frame, tiempo_transcurrido):
    radio_reloj = 50
    centro_reloj = (frame.shape[1] - 150, 100)
    color_reloj = (0, 0, 0)
    cv2.circle(frame, centro_reloj, radio_reloj, color_reloj, 2)

    angulo = 360 * (tiempo_transcurrido % 60) / 60 - 90
    x = int(centro_reloj[0] + radio_reloj * 0.9 * np.cos(np.radians(angulo)))
    y = int(centro_reloj[1] + radio_reloj * 0.9 * np.sin(np.radians(angulo)))
    cv2.line(frame, centro_reloj, (x, y), color_reloj, 2)

    angulo = 360 * ((tiempo_transcurrido // 60) % 60) / 60 - 90
    x = int(centro_reloj[0] + radio_reloj * 0.7 * np.cos(np.radians(angulo)))
    y = int(centro_reloj[1] + radio_reloj * 0.7 * np.sin(np.radians(angulo)))
    cv2.line(frame, centro_reloj, (x, y), color_reloj, 2)

def calcular_angulo(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radianes = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angulo = np.abs(radianes * 180.0 / np.pi)
    if angulo > 180.0:
        angulo = 360 - angulo
    return angulo

def dibujar_barra_progreso_flexiones(imagen, progreso):
    alto, ancho, _ = imagen.shape
    alto_barra = int(alto * 0.5)
    ancho_barra = 15
    x_barra = 20
    y_barra = int((alto - alto_barra) / 4)
    cv2.rectangle(imagen, (x_barra, y_barra), (x_barra + ancho_barra, y_barra + alto_barra), (200, 200, 200), -1)
    alto_lleno = int(alto_barra * progreso)
    cv2.rectangle(imagen, (x_barra, y_barra + (alto_barra - alto_lleno)), (x_barra + ancho_barra, y_barra + alto_barra), (255, 0, 0), -1)
    texto = f'{int(progreso * 100)}%'
    tamaño_texto = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    x_texto = x_barra + (ancho_barra - tamaño_texto[0]) // 2
    y_texto = y_barra - 10
    cv2.rectangle(imagen, (x_texto - 5, y_texto - tamaño_texto[1] - 5), (x_texto + tamaño_texto[0] + 5, y_texto + 5), (255, 0, 0), -1)
    cv2.putText(imagen, texto, (x_texto, y_texto), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def dibujar_reloj_flexiones(imagen, tiempo_inicio):
    tiempo_transcurrido = datetime.now() - tiempo_inicio
    segundos_transcurridos = int(tiempo_transcurrido.total_seconds())
    texto_tiempo = f'tiempo: {segundos_transcurridos}s'
    x_reloj = imagen.shape[1] - 200
    y_reloj = imagen.shape[0] - 20
    cv2.putText(imagen, texto_tiempo, (x_reloj, y_reloj), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

def calcular_retroalimentacion(puntos, posicion, imagen):
    retroalimentacion = []
    advertencias = []

    hombro_izquierdo = [puntos[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * imagen.shape[1], puntos[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * imagen.shape[0]]
    codo_izquierdo = [puntos[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * imagen.shape[1], puntos[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * imagen.shape[0]]
    muñeca_izquierda = [puntos[mp_pose.PoseLandmark.LEFT_WRIST.value].x * imagen.shape[1], puntos[mp_pose.PoseLandmark.LEFT_WRIST.value].y * imagen.shape[0]]
    hombro_derecho = [puntos[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * imagen.shape[1], puntos[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * imagen.shape[0]]
    codo_derecho = [puntos[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * imagen.shape[1], puntos[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * imagen.shape[0]]
    muñeca_derecha = [puntos[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * imagen.shape[1], puntos[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * imagen.shape[0]]
    cadera_izquierda = [puntos[mp_pose.PoseLandmark.LEFT_HIP.value].x * imagen.shape[1], puntos[mp_pose.PoseLandmark.LEFT_HIP.value].y * imagen.shape[0]]
    cadera_derecha = [puntos[mp_pose.PoseLandmark.RIGHT_HIP.value].x * imagen.shape[1], puntos[mp_pose.PoseLandmark.RIGHT_HIP.value].y * imagen.shape[0]]
    rodilla_izquierda = [puntos[mp_pose.PoseLandmark.LEFT_KNEE.value].x * imagen.shape[1], puntos[mp_pose.PoseLandmark.LEFT_KNEE.value].y * imagen.shape[0]]
    rodilla_derecha = [puntos[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * imagen.shape[1], puntos[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * imagen.shape[0]]
    tobillo_izquierdo = [puntos[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * imagen.shape[1], puntos[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * imagen.shape[0]]
    tobillo_derecho = [puntos[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * imagen.shape[1], puntos[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * imagen.shape[0]]

    # Calcular ángulos
    hombro_cadera_tobillo_izquierda = calcular_angulo(hombro_izquierdo, cadera_izquierda, tobillo_izquierdo)
    hombro_cadera_tobillo_derecha = calcular_angulo(hombro_derecho, cadera_derecha, tobillo_derecho)
    codo_hombro_cadera_izquierda = calcular_angulo(codo_izquierdo, hombro_izquierdo, cadera_izquierda)
    codo_hombro_cadera_derecha = calcular_angulo(codo_derecho, hombro_derecho, cadera_derecha)
    angulo_rodilla_cadera_izquierda = calcular_angulo(rodilla_izquierda, cadera_izquierda, hombro_izquierdo)
    angulo_rodilla_cadera_derecha = calcular_angulo(rodilla_derecha, cadera_derecha, hombro_derecho)
    hombro_codo_izquierda = calcular_angulo(hombro_izquierdo, codo_izquierdo, muñeca_izquierda)
    hombro_codo_derecha = calcular_angulo(hombro_derecho, codo_derecho, muñeca_derecha)

    # comentarios basados en los ángulos
    if posicion == "arriba":
        if 160 < hombro_cadera_tobillo_izquierda < 180 and 160 < hombro_cadera_tobillo_derecha < 180:
            retroalimentacion.append("Excelente alineacion de la espalda y los hombros!")
        else:
            advertencias.append("Alineacion de espalda y hombros incorrecta!")
        if 70 < codo_hombro_cadera_izquierda < 100 and 70 < codo_hombro_cadera_derecha < 100:
            retroalimentacion.append("Buena posicion de los codos, manteniendolos cerca del cuerpo!")
        else:
            advertencias.append("Alineacion de los codos incorrecta!")
        if 160 < hombro_cadera_tobillo_izquierda < 180 and 160 < hombro_cadera_tobillo_derecha < 180:
            retroalimentacion.append("Caderas alineadas correctamente!")
        else:
            advertencias.append("Caderas desalineadas!")
        if 80 < hombro_codo_izquierda < 170 and 80 < hombro_codo_derecha < 170:
            retroalimentacion.append("Buena posicion de los brazos durante el ascenso!")
        else:
            advertencias.append("Posicion de los brazos incorrecta durante el ascenso!")
        if 30 < angulo_rodilla_cadera_izquierda < 150 and 30 < angulo_rodilla_cadera_derecha < 150:
            retroalimentacion.append("Buena flexion de las rodillas durante el ascenso!")
        else:
            advertencias.append("Flexion de las rodillas incorrecta durante el ascenso!")

    elif posicion == "abajo":
        if codo_hombro_cadera_izquierda < 90 and codo_hombro_cadera_derecha < 90:
            retroalimentacion.append("Buena profundidad en la flexion!")
        else:
            advertencias.append("Profundidad en la flexion insuficiente!")
        if 70 < codo_hombro_cadera_izquierda < 100 and 70 < codo_hombro_cadera_derecha < 100:
            retroalimentacion.append("Codos bien alineados, manteniendolos cerca del cuerpo!")
        else:
            advertencias.append("Alineacion de los codos incorrecta!")
        if 80 < hombro_cadera_tobillo_izquierda < 100 and 80 < hombro_cadera_tobillo_derecha < 100:
            retroalimentacion.append("Buena alineacion del cuerpo durante el descenso!")
        else:
            advertencias.append("Alineacion del cuerpo incorrecta durante el descenso!")

    return retroalimentacion, advertencias

def mostrar_retroalimentacion(imagen, retroalimentacion, advertencias, precision, fase):
    overlay = imagen.copy()
    alpha = 0.6  
    cv2.rectangle(overlay, (0, 0), (imagen.shape[1], imagen.shape[0]), (0, 0, 0), -1)
    imagen = cv2.addWeighted(overlay, alpha, imagen, 1 - alpha, 0)
    texto = f'Precision {fase}: {precision:.2f}%'
    tamaño_texto, _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
    x_texto = (imagen.shape[1] - tamaño_texto[0]) // 2
    y_texto = (imagen.shape[0] + tamaño_texto[1]) // 2
    cv2.putText(imagen, texto, (x_texto, y_texto), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)

    y_retroalimentacion = 40  
    for i, msg in enumerate(retroalimentacion):
        cv2.putText(imagen, msg, (300, y_retroalimentacion + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    y_advertencias = y_retroalimentacion + 30 * len(retroalimentacion)  
    for i, msg in enumerate(advertencias):
        cv2.putText(imagen, msg, (300, y_advertencias + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    return imagen

# Función para procesar video de sentadillas
def procesar_video_sentadillas(video_path, window_name):
    mp_dibujo = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)

    historial_angulos = []  
    img_grafica = None  
    mostrar_precision = False  
    tiempo_mostrar_precision = 0  
    ultima_precision = 0  
    ultimas_advertencias = []  
    ultima_retroalimentacion_positiva = []  
    tiempo_inicio = time.time()  

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        contador_repeticiones = 0
        posicion = None
        ultimo_estado = 3
        arriba = False
        abajo = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (1024, 600))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False

            lm = pose.process(frame).pose_landmarks
            if not lm:
                continue

            lm_arr = lm.landmark
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if ultimas_advertencias:
                color_conexiones = (0, 0, 255)
            else:
                color_conexiones = (0, 255, 0)

            mp_dibujo.draw_landmarks(frame, lm, mp_pose.POSE_CONNECTIONS,
                                     mp_dibujo.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                     mp_dibujo.DrawingSpec(color=color_conexiones, thickness=2, circle_radius=2))

            angulo_derecho = encontrar_angulo(lm_arr[24], lm_arr[26], lm_arr[28])
            coords_rodilla = (int(lm_arr[26].x * frame.shape[1]), int(lm_arr[26].y * frame.shape[0]))
            cv2.putText(frame, f'{int(angulo_derecho)}', coords_rodilla, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            estado_derecho = estado_pierna(angulo_derecho)

            if estado_derecho == 1:
                if not abajo:
                    abajo = True
                    contador_repeticiones += 0.5  
                    ultima_precision, ultimas_advertencias, ultima_retroalimentacion_positiva = calcular_precision_sentadilla(lm_arr, 'abajo')
                    mostrar_precision = True
                    tiempo_mostrar_precision = cv2.getTickCount()
                posicion = "abajo"
            elif estado_derecho == 3:
                if abajo:
                    arriba = True
                    contador_repeticiones += 0.5
                    ultima_precision, ultimas_advertencias, ultima_retroalimentacion_positiva = calcular_precision_sentadilla(lm_arr, 'arriba')
                    mostrar_precision = True
                    tiempo_mostrar_precision = cv2.getTickCount() 
                    abajo = False
                posicion = "arriba"

            progreso = (angulo_derecho - 60) / (150 - 60)
            progreso = max(0, min(progreso, 1))
            dibujar_barra_progreso(frame, progreso, 950, 50, 30, 500)

            historial_angulos.append(angulo_derecho)

            if len(historial_angulos) % 10 == 0 and len(historial_angulos) > 1:
                fig, ax = plt.subplots()
                ax.plot(historial_angulos, color='blue')
                ax.set_title('Ángulos de Sentadilla')
                ax.set_xlabel('Frames')
                ax.set_ylabel('Ángulo')
                ax.set_facecolor('gray')
                plt.savefig('graph.png', bbox_inches='tight')
                plt.close(fig)
                
                img_grafica = cv2.imread('graph.png')
                img_grafica = cv2.resize(img_grafica, (300, 200))

            if img_grafica is not None:
                frame[-210:-10, 10:310] = img_grafica

            cv2.rectangle(frame, (0, 0), (250, 73), (245, 110, 16), -1)
            cv2.putText(frame, 'CONTADOR-ETAPA', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, f'{int(contador_repeticiones)}{posicion}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 2, cv2.LINE_AA)

            if mostrar_precision:
                tiempo_transcurrido = (cv2.getTickCount() - tiempo_mostrar_precision) / cv2.getTickFrequency()
                if tiempo_transcurrido < 2:
                    overlay = frame.copy()
                    alpha = 0.6  
                    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
                    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                    texto = f'Precision: {ultima_precision:.2f}%'
                    tamano_texto, _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
                    x_texto = (frame.shape[1] - tamano_texto[0]) // 2
                    y_texto = (frame.shape[0] + tamano_texto[1]) // 2
                    cv2.putText(frame, texto, (x_texto, y_texto), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)

                    for i, retro in enumerate(ultima_retroalimentacion_positiva):
                        cv2.putText(frame, retro, (10, 150 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                    for i, advertencia in enumerate(ultimas_advertencias):
                        cv2.putText(frame, advertencia, (10, 150 + 30 * (i + len(ultima_retroalimentacion_positiva))), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    mostrar_precision = False 

            tiempo_transcurrido = time.time() - tiempo_inicio
            dibujar_reloj(frame, tiempo_transcurrido)

            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()

# Función para procesar video de flexiones
def procesar_video_flexiones(video_path, window_name):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)

    contador_frames = 0
    contador = 0
    posicion = None
    progreso = 0
    historial_angulos = []
    img_grafico = None
    ultimos_puntos = None

    mostrar_precision = False
    tiempo_mostrar_precision = 0
    precision_descendente = 0
    precision_ascendente = 0
    retroalimentacion_descendente = []
    retroalimentacion_ascendente = []
    advertencias_descendentes = []
    advertencias_ascendentes = []

    tiempo_inicio = datetime.now()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            imagen = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imagen.flags.writeable = False

            results = pose.process(imagen)

            imagen.flags.writeable = True
            imagen = cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                ultimos_puntos = results.pose_landmarks

            if ultimos_puntos:
                mp_drawing.draw_landmarks(
                    imagen,
                    ultimos_puntos,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=5),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=5)
                )
                try:
                    puntos = ultimos_puntos.landmark

                    hombro_izquierdo = [puntos[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * imagen.shape[1], puntos[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * imagen.shape[0]]
                    codo_izquierdo = [puntos[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * imagen.shape[1], puntos[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * imagen.shape[0]]
                    muñeca_izquierda = [puntos[mp_pose.PoseLandmark.LEFT_WRIST.value].x * imagen.shape[1], puntos[mp_pose.PoseLandmark.LEFT_WRIST.value].y * imagen.shape[0]]
                    hombro_derecho = [puntos[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * imagen.shape[1], puntos[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * imagen.shape[0]]
                    codo_derecho = [puntos[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * imagen.shape[1], puntos[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * imagen.shape[0]]
                    muñeca_derecha = [puntos[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * imagen.shape[1], puntos[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * imagen.shape[0]]
                    cadera_izquierda = [puntos[mp_pose.PoseLandmark.LEFT_HIP.value].x * imagen.shape[1], puntos[mp_pose.PoseLandmark.LEFT_HIP.value].y * imagen.shape[0]]
                    cadera_derecha = [puntos[mp_pose.PoseLandmark.RIGHT_HIP.value].x * imagen.shape[1], puntos[mp_pose.PoseLandmark.RIGHT_HIP.value].y * imagen.shape[0]]
                    rodilla_izquierda = [puntos[mp_pose.PoseLandmark.LEFT_KNEE.value].x * imagen.shape[1], puntos[mp_pose.PoseLandmark.LEFT_KNEE.value].y * imagen.shape[0]]
                    rodilla_derecha = [puntos[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * imagen.shape[1], puntos[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * imagen.shape[0]]
                    tobillo_izquierdo = [puntos[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * imagen.shape[1], puntos[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * imagen.shape[0]]
                    tobillo_derecho = [puntos[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * imagen.shape[1], puntos[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * imagen.shape[0]]

                    angulo = calcular_angulo(hombro_derecho, codo_derecho, muñeca_derecha)

                    historial_angulos.append(angulo)

                    cv2.putText(imagen, f'Elbow Angle: {int(angulo)}', (int(codo_derecho[0]), int(codo_derecho[1] - 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                    if angulo > 160:
                        posicion = "arriba"
                        progreso = 0
                    elif angulo < 90:
                        progreso = 1
                        if posicion == 'arriba':
                            posicion = "abajo"
                            contador += 1
                            retroalimentacion_descendente, advertencias_descendentes = calcular_retroalimentacion(puntos, 'abajo', imagen)
                            precision_descendente = (3 - len(advertencias_descendentes)) / 3 * 100
                            mostrar_precision = True
                            tiempo_mostrar_precision = cv2.getTickCount()
                    else:
                        progreso = (160 - angulo) / (160 - 90)

                    if posicion == "abajo":
                        retroalimentacion_ascendente, advertencias_ascendentes = calcular_retroalimentacion(puntos, 'arriba', imagen)
                        precision_ascendente = (5 - len(advertencias_ascendentes)) / 5 * 100
                        mostrar_precision = True
                        tiempo_mostrar_precision = cv2.getTickCount()

                except Exception as e:
                    print(f"Error: {e}")
                    pass

                cv2.rectangle(imagen, (0, 0), (250, 73), (245, 110, 16), -1)
                cv2.putText(imagen, 'CONTADOR-ETAPA', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(imagen, f'{contador}{posicion}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 2, cv2.LINE_AA)

                dibujar_barra_progreso_flexiones(imagen, progreso)

                if contador_frames % 10 == 0 and len(historial_angulos) > 1:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.plot(historial_angulos, color='blue')
                    ax.set_title('Angulos de Flexion')
                    ax.set_xlabel('Frames')
                    ax.set_ylabel('Angulo')
                    ax.set_facecolor('gray')
                    plt.savefig('graph.png', bbox_inches='tight')
                    plt.close(fig)

                    img_grafico = cv2.imread('graph.png')
                    img_grafico = cv2.resize(img_grafico, (400, 300))
                if img_grafico is not None:
                    imagen[10:310, -410:-10] = img_grafico

                if mostrar_precision:
                    tiempo_transcurrido = (cv2.getTickCount() - tiempo_mostrar_precision) / cv2.getTickFrequency()
                    if tiempo_transcurrido < 2:
                        if posicion == "abajo":
                            imagen = mostrar_retroalimentacion(imagen, retroalimentacion_descendente, advertencias_descendentes, precision_descendente, "Descendente")
                        else:
                            imagen = mostrar_retroalimentacion(imagen, retroalimentacion_ascendente, advertencias_ascendentes, precision_ascendente, "Ascendente")
                    else:
                        mostrar_precision = False

                dibujar_reloj_flexiones(imagen, tiempo_inicio)

            cv2.imshow(window_name, imagen)

            contador_frames += 1

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()

if __name__ == "__main__":
    cv2.namedWindow('Video Processing', cv2.WINDOW_NORMAL)
    procesar_video_sentadillas(r"C:\Users\DELL\Downloads\video123.mp4", 'Video Processing')
    procesar_video_flexiones(r"C:\Users\DELL\Downloads\video2000.mov", 'Video Processing')
    cv2.destroyAllWindows()
