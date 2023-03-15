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
from train import model
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
timerundrow = time.time()
timerundis = time.time()
emailbasetime = time.time()
emailsendtime = time.time()

face_cascade = cv2.CascadeClassifier(current_dir + '/haar_models/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(current_dir + '/haar_models/haarcascade_eye.xml')
MyModel = "model_18_64.t7"  # 定义训练好的人眼状态检测模型的文件名。

# 定义人眼状态检测模型的输入图像大小和输出类别。
shape = (24, 24)
classes = [
    'Close',
    'Open',
]

# 定义一些全局变量，包括检测到的眼睛、检测到的人脸数目和当前的人眼状态。
eyess = []
cface = 0
state = ""


# 定义一个函数，将RGB888格式的图像转换为RGB565格式。
def RGB888toRGB565(img):
    img = img.astype(np.uint8)
    temp = (img[:, :, 0] >> 5) << 5 | (img[:, :, 1] >> 5) << 2 | (img[:, :, 2] >> 6)
    return temp


# 定义一个预处理函数，它将图像路径作为输入，并使用OpenCV和人脸识别算法检测出输入图像中的人脸和眼睛。
def preprocess(image_path):
    timebasedrow = time.time()
    global cface  # 定义一个图像预处理变换，用于将图像转换为PyTorch张量格式。
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])
    image = cv2.imread(image_path['path'])  # 读取图像文件并将其作为OpenCV图像对象加载到内存中。
    faces = face_cascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(1, 1),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    if len(faces) == 0:  # 如果没有检测到任何人脸，则返回空。
        ...
    else:
        cface = 1  # 如果检测到人脸，则设置全局变量cface为1，表示有人脸被检测到。
        (x, y, w, h) = faces[0]  # 获取检测到的第一个人脸的位置和大小。
        face = image[y:y + h, x:x + w]  # 从原始图像中提取人脸区域。
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)  # 在原始图像中绘制一个矩形来标记检测到的人脸。
        roi_color = image[y:y + h, x:x + w]
        """
        Depending on the quality of your camera, this number can vary 
        between 10 and 40, since this is the "sensitivity" to detect the eyes.
        """
        # 在人脸区域中检测眼睛。
        sensi = 20  # 灵敏度, 数字越大越容易检测到眼睛
        eyes = eye_cascade.detectMultiScale(face, 1.3, sensi)
        i = 0
        for (ex, ey, ew, eh) in eyes:
            (x, y, w, h) = eyes[i]
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)  # 在人脸区域中绘制一个矩形来标记检测到的眼睛。
            eye = face[y:y + h, x:x + w]  # 从人脸区域中提取眼睛区域。
            eye = cv2.resize(eye, shape)  # 调整眼睛区域的大小为(24,24)。
            eyess.append([transform_test(Image.fromarray(eye).convert('L')), eye,
                          cv2.resize(face, (48, 48))])  # 将眼睛区域转换为PyTorch张量格式，并存储到eyess列表中。
            i = i + 1
    cv2.imwrite(current_dir + '/temp-images/display.jpg', image)  # 将标记了人脸和眼睛的图像保存到临时目录中。


def eye_status(image, name, net):  # 该函数接受三个参数，一个图像（image），一个名称（name）和一个神经网络（net）。
    img = torch.stack([image[name]])  # 将输入的图像的名称参数作为索引来获取图像，然后将其放入张量中。这个张量叫做img。

    # 使用bp对象和img张量来计算前向传播。它返回两个值：一个是probs，表示预测的概率；另一个是ids，表示每个预测对应的id。
    bp = BackPropagation(model=net)
    probs, ids = bp.forward(img)

    # 获取预测id的第一个元素作为actual_status，然后将预测概率的第一个元素赋给prob。如果actual_status等于0，则将预测概率的第二个元素赋给prob。
    actual_status = ids[:, 0]
    prob = probs.data[:, 0]
    if actual_status == 0:
        prob = probs.data[:, 1]

    # print(name,classes[actual_status.data], probs.data[:,0] * 100)
    return classes[actual_status.data]  # 返回在classes列表中actual_status索引处的元素。


# 该函数接受两个参数，一个图像(imag)和一个模型名称(modl)。它使用imag参数生成一个字典，该字典包含一个图像的路径和一个空的眼睛元组。
# 然后，它调用drow的函数，将这个字典作为参数传递给它，并将模型名称作为model_name参数传递给它。
def func(imag, modl):
    drow(images=[{'path': imag, 'eye': (0, 0, 0, 0)}], model_name=modl)


# 该函数接受两个参数，一个包含字典的列表(images)和一个模型名称(model_name)。
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
    net = model.Model(num_classes=len(classes))
    checkpoint = torch.load(os.path.join(current_dir + '/model', model_name), map_location=device)  # 根据类别数量创建一个神经网络对象。

    ##从磁盘中加载训练过的模型并将其应用到net对象上。
    net.load_state_dict(checkpoint['net'])
    net.eval()
    # 初始化flag和status变量。
    flag = 1
    status = ""
    for i, image in enumerate(images):  # 对于每张图像，检查flag的值。如果它是1，则调用preprocess函数，该函数对输入的图像进行预处理，并将flag设置为0。
        if (flag):
            preprocess(image)
            flag = 0

        # 如果未检测到人脸（cface==0），则使用OpenCV加载一张预定义的图像，其中包含“未检测到人脸”的文本，并将其写入temp-images/display.jpg。
        # 同时将timebasedrow、timebasedis、timerundrow和timerundis的值设置为当前时间。
        if cface == 0:
            image = cv2.imread(current_dir + "/temp-images/display.jpg")
            image = cv2.putText(image, 'No face Detected', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 0, 0), 20,
                                cv2.LINE_AA)
            image = cv2.putText(image, 'No face Detected', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 6, (255, 255, 255), 10,
                                cv2.LINE_AA)
            cv2.imwrite(current_dir + '/temp-images/display.jpg', image)
            # 获取当前时间戳，并将其存储在变量timebasedrow、timebasedis、timerundrow和timerundis中。
            timebasedrow = time.time()
            timebasedis = time.time()
            timerundrow = time.time()
            timerundis = time.time()
            print('未检测到人脸')

        # 如果检测到眼睛（len(eyess)!=0），则获取第i张图像中的眼睛和人脸信息，并将其保存在image变量中。然后使用eye_status函数检查眼睛的状态。如果眼睛闭上了，将timerundis的值设置为当前时间，并检查自从
        # timebasedis以来是否已经超过1.5秒。如果是，则使用OpenCV加载一张预定义的图像，其中包含“分心”的文本，并将其写入temp-images/display.jpg。如果音频没有播放，则开始播放。
        elif (len(eyess) != 0):
            print('检测到人脸，并识别到眼睛状态')
            eye, eye_raw, face = eyess[i]
            image['eye'] = eye
            image['raw'] = eye_raw
            image['face'] = face
            timebasedrow = time.time()
            timerundrow = time.time()
            for index, image in enumerate(images):
                status = eye_status(image, 'eye', net)
                if (status == "Close"):
                    print('检测到用户闭眼')
                    timerundis = time.time()
                    if ((timerundis - timebasedis) > 1.5):
                        print('用户状态警告，闭眼时间为:' + str(timerundis - timebasedis) + ',当前时间为:' + now_time)
                        image = cv2.imread(current_dir + '/temp-images/display.jpg')
                        state = "Distracted"
                        image = cv2.putText(image, 'Distracted', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 7, (0, 0, 0), 20,
                                            cv2.LINE_AA)
                        image = cv2.putText(image, 'Distracted', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 7,
                                            (255, 255, 255), 10, cv2.LINE_AA)
                        cv2.imwrite(current_dir + '/temp-images/display.jpg', image)

                        # print('Distracted end'+now_time)
                        if (not (p.is_playing())):
                            print('开始播放音频')
                            p.play()
                else:
                    print('用户状态良好，停止播放音频')
                    p.stop()  # 如果眼睛未闭合，则停止播放音频。

        # 如果没有检测到人脸或眼睛，则检查自从timebasedrow以来是否已经超过3秒。如果是，则使用OpenCV加载一张预定义的图像，其中包含“昏昏欲睡”的文本，
        # 并将其写入temp-images/display.jpg。如果音频没有播放，则开始播放。同时将state设置为“Drowsy”（昏昏欲睡）。
        else:
            print('无眼睛数据,开始判断用户状态')
            timerundrow = time.time()
            # print('timerundrow'+str(timerundrow))
            # print('timebasedrow'+str(timebasedrow))
            print('-------------------------------')
            if ((timerundrow - timebasedrow) > 5):
                print('用户闭眼:' + str(timerundrow - timebasedrow) + 's，判定为‘危险驾驶’，现在时刻为:' + now_time)
                # print('Drowsy start run' + str(timerundrow))
                # print('Drowsy start base' + str(timebasedrow))
                # print('*********************************')
                if (not (p.is_playing())):
                    print('开始播放音频')
                    p.play()
                image = cv2.imread(current_dir + '/temp-images/display.jpg')
                state = "Drowsy"
                image = cv2.putText(image, 'Drowsy', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 7, (0, 0, 0), 20, cv2.LINE_AA)
                image = cv2.putText(image, 'Drowsy', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 7, (255, 255, 255), 10,
                                    cv2.LINE_AA)
                cv2.imwrite(current_dir + '/temp-images/display.jpg', image)
                emailsendtime = time.time()
                if (emailsendtime - emailbasetime > 60 or num == 0):
                    num += 1
                    location_get()
                    dataJson = json.load(open('/home/qinghuan/project/AI/yolo/Driving-Guardian/jetson Code/Drowsiness/json.json', encoding='UTF-8'))  # 打开json文件，并将其中的数据全部读取
                    print('距离上次发送邮件已经:' + str(emailsendtime - emailbasetime) + 's，发送邮件')
                    addr = current_dir + '/report/message.txt'
                    with open(addr, 'w+', encoding='utf-8') as message:
                        message.truncate(0)
                        message.write('用户在' + now_time + '时存在危险驾驶行为'+'\n'+'位置信息为:'+str(dataJson))
                    SendEmail().send_email(addr)  # 发送邮件
                    emailbasetime = time.time()
                timebasedrow = time.time()
                # print('Drowsy end' + now_time)


video_capture = cv2.VideoCapture(1)

if __name__ == '__main__':
    try:
        emailbasetime = time.time()
        num = 0
        while 1:
            p.play()
            eyess = []
            cface = 0
            state = ""
            ret, img = video_capture.read()  # 从相机读取一帧，返回值 ret 为 True 表示成功读取帧，img 为图像数据。
            cv2.imwrite(current_dir + '/temp-images/img.jpg', img)
            func(current_dir + '/temp-images/img.jpg', MyModel)
            img = cv2.imread(current_dir + '/temp-images/display.jpg')
            w = 240
            h = 180
            dim = (w, h)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            img = RGB888toRGB565(img)
            img = img.flatten()


    except:
        video_capture.release()
        print("\nGoodbye")

