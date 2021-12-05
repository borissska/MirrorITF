import time
import psycopg2
from cv2 import cv2
import numpy as np
import mediapipe as mp
import time

# Подключение
# Подключение к картинке
# img = cv.imread('filename.jpg') - импортировать картинку
# cv.imshow('Name', file) - показать картинку

# Подключение к видео
# capture = cv.VideoCapture(0) - считывание трансляции с видеокамеры
# (0, 1, 2) - номер видеокамеры подключенной к компу
# ('filename.mp4') - путь к файлу на компьютере

# Работа с видео
# while True:
#   isTrue - лигическая переменная, которая говорит, считалась ли картинка,
#   frame = capture.read() - считывание информации с видео, картинку за картинкой
#   cv.imshow('Name', frame) - вывод картинок на экран (видео)
# capture.release() - прекращаем видеосъемку
# cv.destroyAllWindows() - закрываем все окна программы

# Изменение размеров
# Изменение картинки, видео, реального видео
# def rescaleFrame(frame, scale = 0.75): - функция изменения размеров передаваемого изображения
#   width = int(frame.shape[1] * scale) - изменение ширины картинки в scale раз
#   height = int(frame.shape[0] * scale) - изменение высоты картинки в scale раз
#   int(something) - явное приведение в целочисленному типу int
#   dimensions = (width, height) - создаем переменную содержащую размер
#   return cv.resize(frame, dimensions, interpolation = cv.INTER_AREA) -
#
#   frame_resized = rescaleFrame(frame, 0.2) - вызов функции изменения размера картинки

# Изменение реального видео
# def changeRes(width, height): - работаем со всем этим через функцию с параметрами width и height
#   capture.set(3, width) - изменяем ширину картинки, 3 - номерпараматра картинки принадлежащий ширине
#   capture.set(4, height) - изменяем высоту картинки, 4 - номерпараматра картинки принадлежащий высоте

# Рисование на картинке
# Создание пустого поля для рисования
# blank = np.zeros((500, 500, 3), dtype = 'uint8') - создание пустого изображения при помощи numpy
# (500, 500, 3) - высота, ширина, количество цветовых каналов (BGR)
# blank[:] = 0, 255, 0 - делаем все пиксели на картинке зелеными
# blank[200:300, 300:400] = 0, 255, 0 - выделяем квадрат 100 на 100 пикселей на картинке

# Рисование прямоугольника
# cv.rectangle(blank, (0, 0), (250, 250), (0, 255, 0), thickness = 2) - выделяет прямойгольную зону на картинке
# (0, 0) - начало прямоугольника
# (250, 250) - конец; (blank.shape[1], blank.shape[0]) - конец рамки равен концу окна
# (0, 255, 0) - цвет
# thickness - толщина рамки (идет вовнутрь); thickness = cv.FILLED = -1 - заполняет всю область внутри прямоугольника

# Рисование круга
# cv.circle(blank, center(blank.shape[1]//2, blank.shape[0]//2), radius(40), color(0, 0, 255), thickness(3))
# - созание круга по заданному центру и радиусу, с нужным цветом и шириной контура

# Рисование линии
# cv.line(blank, (0, 0), (blank.shape[1]//2, blank.shape[0]//2), (0, 255, 0), thickness = 2)
# - рисование лини с заданными координатами начала, конци, цвета и ширины линии

# Написание текста на картинке
# cv.putText(blank, 'Hello', (100, 100), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 0), 2)
# - вывод картинки с определенными словами, точкой отсчета, типом текста, размером, цветом и жирностью текста

# Основные функции
# Перевод изображения в черно-белые тона
# frame_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) - изменение цветной картинки на картинку с оттенками серого

# Размытие изображения
# blur = cv.GaussianBlur(img, (3, 3), cv.BORDER_DEFAULT) - размытие изображения с превращением 9 пикселей в 1

# Создание окантовки
# canny = cv.Canny(blur, 125, 175) - создание окантовки

# Расширение картинки
# dilated = cv.dilate(canny, (3, 3), iterations = 1) - увеличивает резкость картинки, повышая количество пикселей i раз

# Уменьшение резкости картинки
# eroded = cv.erode(dilated, (3, 3), iterations = 1) - увеличивает мягкость картинки по 9 пикселей i раз

# Изменение размеров картинки
# resized = cv.resize(img, (500, 500), interpolation = cv.INTER_CUBIC)
# - изменение размеров изображения: сжание/расширение. Интерполяция - подстановка неизвестных значений

# Обрезка изображения
# cropped = img[100:200, 200:300] - вырезание определенного куска изображения

# Азы изменения расположения картинки
# Перемещение изображения
# def translate(img, x, y):
#   transMat = np.float32([[1, 0, x], [0, 1, y]]) - передвинутая матрица значений цветов
#   dimensions = (img.shape[1], img.shape[0]) - создаем переменную содержащую размер
#   return cv.warpAffine(img, transMat, dimensions) - функция возвращения от значений к цветам
# translated = translate(imd, 100, 100) - сдвиг изображения на 100 вправо и 100 вниз
# -x --> Лево
# -y --> Вверх
# x --> Вправо
# y --> Вниз

# Поворот изображения
# def rotate(img, angle, rotPoint = None):
#   (height, width) = img.shape[:2] - приравнивание высоты и ширины первым двум параметрам картинки
#   if rotPoint is None:
#       rotPoint = (width//2, height//2) - точка относительно которой будет происходить вращение
#   rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
#   - производит поворот картинки относительно центра, на угол, без изменений размерности
#   dimensions = (width, height) - создаем переменную содержащую размер
#   return cv.warpAffine(img, rotMat, dimensions) - функция возвращения от значений к цветам
# rotated = rotate(imd, 45) - поворот изображения на 45 градусов против часовой стрелки

# Изменение размера изображения
# resized = cv.resize(img, (500, 500), interpolation = cv.INTER_CUBIC)
# - изменение размеров изображения: сжание/расширение. Интерполяция - подстановка неизвестных значений
# INTER_AREA, INTER_DEFAULT - уменьшение; INTER_CUBIC, INTER_LINEAR - увеличение

# Переворот изображения
# flip = cv.flip(img, 0) - отражение картинки (0 - горизонтальное; 1 - вертикальное; -1 - вертикально и горизонтально)

# Выделение контура
# gray = cv.cvtColor(img, cv.BGR2GRAY) - перевод цветного изображения в оттенки серого
# blur = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT) - размытие изображения с превращением 9 пикселей в 1
# canny = cv.Canny(blur, 125, 175) - создание окантовки
# ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY) - приведение цветов к белым и черным
# contours, hierarchies = cv.findContours(canny/thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
# - метод, который выделяет контуры на картинке
# RETR_EXTERNAL - внешний контур; RETR_TREE - контуры высокой важности; RETR_LIST - все контуры
# Обработка контура: cv.CHAIN_APPROX_NONE - ничего; cv.CHAIN_APPROX_SIMPLE - упрощает создание контура
# cv.drawContours(blank, contours, -1, (0, 0, 255), 1) - перенос контура на пустое изображение, берем все контуры(-1),
# (0, 0, 255) - красный цвет контура; сохранение в blank

# Работа со временем ожидания
# cv.waitKey(0) - бесконечно ждеп пока нажмут кнопку
# if cv.waitKey(20) & oxFF == ord('q'): - если прошло 20 секунд и нажали кнопку 'q', то
#   break - работа останавливается

conn = psycopg2.connect(dbname='d7ed08ab2v640', user='xhcapgvloicexy',
                        password='276dab641de7c77214bf42fd0afc1f13ce8f4ee025c06b871325767572ecbc9b', host="ec2-34-242-89-204.eu-west-1.compute.amazonaws.com")
cursor = conn.cursor()

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)
# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, smooth_landmarks=True) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor Feed
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Make Detections
        results = holistic.process(img)

        export_data = results.pose_landmarks[0]

        # print(results.face_landmarks)

        # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks

        # Recolor image back to BGR for rendering
        image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # 1. Draw face landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                  mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                                  )

        # 2. Right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                  )

        # 3. Left Hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                  )

        # 4. Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

        cv2.imshow('Raw Webcam Feed', image)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

cursor.close()
conn.close()