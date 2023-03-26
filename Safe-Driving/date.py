import paho.mqtt.client as mqtt

def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    client.subscribe("/iah3wr181JG/pc_get/user/get")

def on_message(client, userdata, msg):
    data = msg.payload.decode()
    # print(msg.payload.decode())
    print(data)


client = mqtt.Client(client_id="iah3wr181JG.pc_get|securemode=2,signmethod=hmacsha256,timestamp=1679801574082|", clean_session=False, userdata=None)
client.username_pw_set("pc_get&iah3wr181JG", "44f8f20517cfef225cb5b6522a50b4cd4aac4397e1de847115e479569732a6dc")
client.on_connect = on_connect
client.on_message = on_message
client.connect("iot-06z009za7jb4890.mqtt.iothub.aliyuncs.com", port=1883, keepalive=60)

client.loop_forever()

