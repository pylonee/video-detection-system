import cv2
import pandas as pd
import argparse
from ultralytics import YOLO

def main ():
    parser = argparse.ArgumentParser() # для чтения из терминала
    parser.add_argument('--video', type=str, required=True, help='file path') # ждем обязательный --video
    args = parser.parse_args() # получаем имя видео которое напишут

    vid = cv2.VideoCapture(args.video) # открываем видео для покадрового чтения
    if not vid.isOpened():
        print("Error: can't open video") # ошибка если видео не открылось
        return

    ret, frame = vid.read() # первый кадр из видео (ret = False - видео повреждено или пустое)
    if not ret:
        print("Error: can't read first frame") # ошибка видео
        return

    roi = cv2.selectROI(frame, showCrosshair=True, fromCenter=False) # даем пользователю выбрать область интереса (стол) на первом кадре
    cv2.destroyAllWindows() # закрываем все окна cv2

    model = YOLO('yolov8n.pt') # модель для поиска людей

    events_log = [] # список для событий
    last_state = 'empty' # последнее состояние для сравнения с текущим
    last_change_time = 0 # время изменения состояния
    empty_time = [] # список для времени от "стол пустой" до "человек подошел"
    frame_count_last_detection = 0 # счетчик сколько кадров стол пустой
    FRAMES_DELAY = 60 # задержка, чтобы подтвердить, что стол пуст (вдруг модель не обнаружит человека на текущем кадре, а он есть. Состояние поменяется - плохо)

    # параметры видео
    fps = vid.get(cv2.CAP_PROP_FPS)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # проверка параметров
    if fps <= 0 or width <= 0 or height <= 0:
        print("Error: invalid video settings")
        vid.release()
        return

    # обработка видео - кодек, кадры в секнуду размер
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_vid = cv2.VideoWriter('output.mp4', fourcc, max(1, int(fps)), (width, height))

    # начинаем обработку с первого кадра
    vid.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        # читаем каждый кадр, когда ret == False - видео закончилось
        ret, frame = vid.read()
        if not ret:
            break

        # координаты области интереса - x,y - левый верхний угол, w,h- ширина и высота
        x, y, w, h = roi
        roi_frame = frame[y:y+h, x:x+w] # обрезаем кадр по области интереса

        res = model(roi_frame, verbose=False, tracker='bytetrack.yaml') # ищем людей на кадре, vrebose=False - не выводить лишнюю инфу

        # записываем сколько людей нашла можель на кадре
        people_count = 0
        if res[0].boxes is not None and res[0].boxes.id is not None:
            people_count = len(res[0].boxes.id) # по id, если id есть

        if res[0].boxes is not None and people_count == 0:
            people_bboxes = res[0].boxes[res[0].boxes.cls == 0]  # либо фильтр на людей (в модели class 0 - person)
            people_count = len(people_bboxes)
        else:
            people_count = 0

        # если люди есть
        if people_count > 0:
            frame_count_last_detection = 0 # сбрасываем счетчик
            current_state = 'occupied' # стол занят
        else:
            frame_count_last_detection += 1 # прибавляем кадры если людей нет (стол пустой)
            if frame_count_last_detection >= FRAMES_DELAY: # задержка в 60 кадров (считаем что стол пустой, если нет людей 60 кадров)
                current_state = 'empty' # пустой стол
            else:
                current_state = 'occupied' # стол занят

        current_time = vid.get(cv2.CAP_PROP_POS_MSEC)/1000.0 # текущее время видео в секундах

        # Если изменилось состояние стола фиксируем изменения
        if current_state != last_state:
            event_type = None
            if current_state == 'occupied' and last_state == 'empty': # изменение с пустой на занят
                event_type = 'step_up' # подошли
                if last_change_time != 0:    # фиксируем время, которое прошло с момента как состояние стало empty
                    pass_time = current_time - last_change_time
                    empty_time.append(pass_time)
            elif current_state == 'empty': # текущее состояние пустой, значит событие пустой
                event_type = 'empty'

            # заполняем строку таблицы
            events_log.append({
                'time': round(current_time, 2),
                'event': event_type,
                'state': current_state,
            })

            # обновляем состояния
            last_state = current_state
            last_change_time = current_time

        # отрисовываем рамку на кадре (красный - стол занят, зеленый - стол пустой)
        color = (0, 255, 0) if current_state == 'empty' else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        cv2.imshow('Video', frame) # показываем кадр (для просмотра сразу)
        output_vid.write(frame)  # записываем кадр в output.mp4

        if cv2.waitKey(1) & 0xFF == ord('q'):  # если нажали q, то выходим цикла (quit)
            break

    vid.release() # закрываем видео
    output_vid.release() # закрываем output
    cv2.destroyAllWindows() # закрываем все окна cv2

    # оздаем таблицу событий + считаем среднее между событиями пустой -> подошли
    df = pd.DataFrame(events_log)
    if len(empty_time) > 0:
        avg_time = sum(empty_time)/len(empty_time)
    else:
        avg_time = 0

    # сохранение csv файла с событиями и статистика в консоли
    print("\n--- Report ---")
    print(f"Total events: {len(df)}")
    print(f"Avg time between departure and approach: {avg_time: .2f} sec")
    df.to_csv('events.csv', index=False)
    print("\n Result saved in output.mp4 and events.csv")

if __name__ == "__main__":
    main()