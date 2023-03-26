import paho.mqtt.client as mqtt
import matplotlib.pyplot as plt
import time

# 配置连接参数
client = mqtt.Client(client_id="iah3wr181JG.pc_get|securemode=2,signmethod=hmacsha256,timestamp=1679801574082|", clean_session=False, userdata=None)
client.username_pw_set("pc_get&iah3wr181JG", "44f8f20517cfef225cb5b6522a50b4cd4aac4397e1de847115e479569732a6dc")
client.connect("iot-06z009za7jb4890.mqtt.iothub.aliyuncs.com", port=1883, keepalive=60)

# 初始化数据
x_data = []
y_data = []

# 定义回调函数，用于处理接收到的消息
def on_message(client, userdata, msg):
    # data = json.loads(message.payload.decode())
    now_time=time.time()
    data = msg.payload.decode()
    x_data.append(now_time)
    y_data.append(data)

def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    client.subscribe("/iah3wr181JG/pc_get/user/get")

# 订阅指定的Topic，并设置回调函数
client.on_connect = on_connect
client.on_message = on_message

# 开始循环，等待接收数据
client.loop_start()

while True:
    # 等待一段时间，接收足够的数据
    time.sleep(30)

    # 绘制折线图
    plt.plot(x_data, y_data)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.show()

# 停止循环
# client.loop_stop()
