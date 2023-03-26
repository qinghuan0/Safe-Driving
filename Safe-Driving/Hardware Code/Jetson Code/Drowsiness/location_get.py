import uuid
import socket
import requests
import requests
import json

def location_get():
    url0 = "http://httpbin.org/ip"  # 也可以直接在浏览器访问这个地址
    r = requests.get(url0)  # 获取返回的值
    ip = json.loads(r.text)["origin"]  # 取其中某个字段的值
    url = f'https://ip.useragentinfo.com/json?ip='+ip
    res = requests.get(url)		# 发送请求
    data = json.loads(res.text)
    with open('./report/json.json','w',encoding='utf-8') as file:
        file.write(json.dumps(data,indent=2,ensure_ascii=False))





