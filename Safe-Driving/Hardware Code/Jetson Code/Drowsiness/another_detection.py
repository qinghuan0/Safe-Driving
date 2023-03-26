import cv2
import torch.hub
import os
import utils_s.model
from PIL import Image
from torchvision import transforms
from utils_s.grad_cam import BackPropagation
import time
import threading
import vlc
import json
import os
import socket
import numpy as np
from train_bed import model
from sent_email import SendEmail
import datetime as dt
from location_get import location_get

current_dir = os.path.dirname(os.path.realpath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file = 'alarm.mp3'
# Sound player start
p = vlc.MediaPlayer(current_dir + '/' + file)  # 音频

# 定义时间戳，用于计算代码执行时间。
now_time = dt.datetime.now().strftime('%F %T')
timebasedrow = time.time()
timebasedis = time.time()
emailbasetime = time.time()
emailsendtime = time.time()
blink_start=time.time()#眨眼时间
blink_count=0#眨眼计数
list_B=np.ones(11)#眼睛状态List,建议根据fps修改

blink_freq=0.5#眨眼频率

face_cascade = cv2.CascadeClassifier(current_dir + '/haar_models/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(current_dir + '/haar_models/haarcascade_eye.xml')
MyModel = "BlinkModel.t7"  # 定义训练好的人眼状态检测模型的文件名。

# 定义人眼状态检测模型的输入图像大小和输出类别。
eye_long=24

shape = (eye_long, eye_long)
classes = [
    'Close',
    'Open',
]

# 定义一些全局变量，包括检测到的眼睛、检测到的人脸数目和当前的人眼状态。
eyess = []
cface = 0
state = ""

# 定义一个预处理函数，它将图像路径作为输入，并使用OpenCV和人脸识别算法检测出输入图像中的人脸和眼睛。
def preprocess(image_path):
    global cface
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])
    image = cv2.imread(image_path['path'])
    faces = face_cascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(1, 1),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    if len(faces) == 0:
        ...
    else:
        cface = 1
        (x, y, w, h) = faces[0]  # 获取检测到的第一个人脸的位置和大小。
        face = image[y:y + h, x:x + w]  # 从原始图像中提取人脸区域。
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)  # 在原始图像中绘制一个矩形来标记检测到的人脸。
        roi_color = image[y:y + h, x:x + w]
        """
        Depending on the quality of your camera, this number can vary 
        between 10 and 40, since this is the "sensitivity" to detect the eyes.
        """
        # 在人脸区域中检测眼睛。
        sensi = 25  # 灵敏度, 数字越大越容易检测到眼睛
        eyes = eye_cascade.detectMultiScale(face, 1.3, sensi)
        i = 0
        for (ex, ey, ew, eh) in eyes:
            (x, y, w, h) = eyes[i]
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)  # 在人脸区域中绘制一个矩形来标记检测到的眼睛。
            eye = face[y:y + h, x:x + w]  # 从人脸区域中提取眼睛区域。
            eye = cv2.resize(eye, shape)  # 调整眼睛区域的大小为(24,24)。
            eyess.append([transform_test(Image.fromarray(eye).convert('L')), eye,
                          cv2.resize(face, (eye_long, eye_long))])  # 将眼睛区域转换为PyTorch张量格式，并存储到eyess列表中。
            i = i + 1
    cv2.imwrite(current_dir + '/temp-images/display.jpg', image)

def eye_status(image, name, net):
    img = torch.stack([image[name]])

    bp = BackPropagation(model=net)
    probs, ids = bp.forward(img)

    # 获取预测id的第一个元素作为actual_status，然后将预测概率的第一个元素赋给prob。如果actual_status等于0，则将预测概率的第二个元素赋给prob。
    actual_status = ids[:, 0]
    prob = probs.data[:, 0]
    if actual_status == 0:
        prob = probs.data[:, 1]

    print(name,classes[actual_status.data], probs.data[:,0] * 100)
    return classes[actual_status.data]  # 返回在classes列表中actual_status索引处的元素。

def func(imag, modl):
    drow(images=[{'path': imag, 'eye': (0, 0, 0, 0)}], model_name=modl)


def drow(images, model_name):
    global eyess
    global cface
    global timebasedrow
    global timebasedis
    global timerundrow
    global timerundis
    global state
    global emailsendtime
    global emailbasetime
    global num
    global blink_start
    global blink_count
    global list_B
    global blink_freq
    net = model.Model(num_classes=len(classes))
    checkpoint = torch.load(os.path.join(current_dir + '/model', model_name), map_location=device)  # 根据类别数量创建一个神经网络对象。

    net.load_state_dict(checkpoint['net'])
    net.eval()

    flag = 1
    status = ""
    for i, image in enumerate(images):  # 对于每张图像，检查flag的值。如果它是1，则调用preprocess函数，该函数对输入的图像进行预处理，并将flag设置为0。
        if (flag):
            preprocess(image)
            flag = 0

        if cface == 0:
            image = cv2.imread(current_dir + "/temp-images/display.jpg")
            image = cv2.putText(image, 'No face Detected', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 0, 0), 20,
                                cv2.LINE_AA)
            image = cv2.putText(image, 'No face Detected', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 6, (255, 255, 255), 10,
                                cv2.LINE_AA)
            cv2.imwrite(current_dir + '/temp-images/display.jpg', image)


        elif (len(eyess) != 0):
            eye, eye_raw, face = eyess[i]
            image['eye'] = eye
            image['raw'] = eye_raw
            image['face'] = face
            timebasedrow = time.time()
            timerundrow = time.time()
            for index, image in enumerate(images):
                status = eye_status(image, 'eye', net)

                if (status == "Close"):
                    list_B = np.append(list_B, 0)  # 闭眼为‘0’

                else:
                     list_B = np.append(list_B, 1)  # 睁眼为‘1’

        else:
            list_B = np.append(list_B, 0)  # 闭眼为‘0’


        list_B = np.delete(list_B, 0)
        # 实时计算PERCLOS
        if cface !=0:
            perclos = 1 - np.average(list_B)
            print('perclos={:f}'.format(perclos))
            if list_B[9] == 1 and list_B[10] == 0 :
                # 如果上一帧为’1‘，此帧为’0‘则判定为眨眼
                print('----------------眨眼----------------------')
                blink_count += 1
            blink_T = time.time() - blink_start
            if blink_T > 10:
                # 每10秒计算一次眨眼频率
                blink_freq = blink_count / blink_T
                blink_start = time.time()
                blink_count = 0
                print('blink_freq={:f}'.format(blink_freq))
            if blink_T > 10:
                # 每10秒计算一次眨眼频率
                blink_freq = blink_count / blink_T
                blink_start = time.time()
                blink_count = 0
                print('blink_freq={:f}'.format(blink_freq))
            if (perclos > 0.4):
                print('疲劳')
                p.play()
            elif (blink_freq < 0.25):
                print('疲劳')
                p.play()
                blink_freq = 0.5  # 如果因为眨眼频率判断疲劳，则初始化眨眼频率
            else:
                print('清醒')
                p.stop()
        else:
            list_B = np.append(list_B, 1)
            print('未检测到人脸')

video_capture = cv2.VideoCapture(0)

WIDTH = 640
HEIGHT = 480

def process_image(image, model):
    cv2.imwrite('temp-images/img.jpg', image)
    func('temp-images/img.jpg', model)
    img = cv2.imread('temp-images/display.jpg')
    img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)

    return img

if __name__ == '__main__':

    emailbasetime = time.time()
    num = 0
    while True:
        start = time.time()
        eyess = []
        cface = 0
        state = ""
        ret, img = video_capture.read()
        if not ret:
            continue
        img = process_image(img, MyModel)
        T = time.time() - start
        fps = 1 / T  # 实时在视频上显示fps
        fps_txt = 'fps:%.2f' % (fps)
        cv2.putText(img,fps_txt,(0,10),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, 8)
        cv2.imshow("start",img)
        if cv2.waitKey(100) & 0xff == ord('q'):
            video_capture.release()
            print("\nGoodbye")
            break
