import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist

# Função para calcular a razão de aspecto do olho
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Constantes
EAR_THRESHOLD = 0.25  # Limiar para considerar como um piscar
EAR_FRAMES = 3        # Número mínimo de frames consecutivos abaixo do limiar

# Carregar detector de rosto e preditor de marcos faciais
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Índices para os marcos faciais dos olhos
(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)

# Inicializar contadores
blink_count = 0
frame_counter = 0

# Captura de vídeo
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Converter para escala de cinza e garantir que seja 8-bit
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Verificar se a imagem está no formato correto
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)
    
    # Detectar rostos
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        shape = np.array([[p.x, p.y] for p in shape.parts()])

        # Coordenadas dos olhos
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        # Calcular EAR
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # Média do EAR dos dois olhos
        ear = (leftEAR + rightEAR) / 2.0

        # Desenhar os contornos dos olhos
        cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)

        # Verificar se o EAR está abaixo do limiar
        if ear < EAR_THRESHOLD:
            frame_counter += 1
        else:
            if frame_counter >= EAR_FRAMES:
                blink_count += 1
            frame_counter = 0

        # Mostrar contagem de piscadas
        cv2.putText(frame, f'Blinks: {blink_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Mostrar o frame com os resultados
    cv2.imshow('Blink Detection', frame)
    
    # Sair com a tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar captura e destruir janelas
cap.release()
cv2.destroyAllWindows()