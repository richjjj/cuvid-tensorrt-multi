import os
import cv2
import json
import threading

def draw(txt_names):
    for txt_name in txt_names:
        f = open("./txt/"+txt_name, "r")
        lines = f.readlines()
        txt_dict = json.loads(lines[0][:-1])
        
        video_path = "/home/xmrbi" + txt_dict["uri"]
        cap = cv2.VideoCapture(video_path)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  #获取视频的宽度
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  #获取视频的高度
        fps = cap.get(cv2.CAP_PROP_FPS) #获取视频的帧率
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        res_video_name = "./result/" + video_path[video_path.rfind("/")+1:]
        writer = cv2.VideoWriter(res_video_name, cv2.VideoWriter_fourcc(*'H264'), fps, (width, height))
        
        draw_frames = []
        draw_boxes = []
        for line in lines:
            dict_tmp = json.loads(line[:-1])
            draw_frames.append(dict_tmp["frame_index"])
            draw_boxes.append(dict_tmp["events"])
    
        frame_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == False or frame_index > frame_count:
                break
            if frame_index in draw_frames:
                events = draw_boxes[draw_frames.index(frame_index)]
                for event in events:
                    for obj in event["objects"]:
                        coordinate = obj["coordinate"]
                        coordinate[0] = max(0, int(coordinate[0]))
                        coordinate[1] = max(0, int(coordinate[1]))
                        coordinate[2] = min(width, int(coordinate[2]))
                        coordinate[3] = min(height, int(coordinate[3]))
                        cv2.rectangle(frame, (coordinate[0], coordinate[1]), (coordinate[2], coordinate[3]), (0, 0, 255), 3)
            writer.write(frame)
            frame_index += 1
            if frame_index % 1000 == 0:
                print(frame_index, " / ", frame_count)
        cap.release()
        writer.release()
        print("done")

txt_names1 = ["10.txt", "3.txt"]
txt_names2 = ["11.txt", "18.txt"]
txt_names3 = ["12.txt", "8.txt"]
txt_names4 = ["24.txt", "4.txt", "9.txt"]

th1 = threading.Thread(target=draw, args=(txt_names1,))
th2 = threading.Thread(target=draw, args=(txt_names2,))
th3 = threading.Thread(target=draw, args=(txt_names3,))
th4 = threading.Thread(target=draw, args=(txt_names4,))
th1.start()
th2.start()
th3.start()
th4.start()
th1.join()
th2.join()
th3.join()
th4.join()
