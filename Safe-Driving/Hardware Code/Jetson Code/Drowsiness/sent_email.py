"""
使用一个邮箱向另一个邮箱发送测试报告的html文件，这里需要对发送邮件的邮箱进行设置，获取邮箱授权码。
username=“发送邮件的邮箱”， password=“邮箱授权码”
这里要特别注意password不是邮箱密码而是邮箱授权码。
mail_server = "发送邮箱的服务器地址"
这里常用的有 qq邮箱——"stmp.qq.com", 163邮箱——"stmp.163.com"
其他邮箱服务器地址可自行百度
"""
import os
import smtplib
from email.mime.text import MIMEText
from email.header import Header
import time

send_email = 'kyzd1484237245@163.com'
reciver_email = '1484237245@qq.com'

# 自动发送邮件
class SendEmail():
    def send_email(self, new_report):
        # 读取测试报告中的内容作为邮件的内容
        with open(new_report, 'r', encoding='utf8') as f:
            mail_body = f.read()
        # 发件人地址
        send_addr = send_email
        # 收件人地址
        reciver_addr = reciver_email
        # 发送邮箱的服务器地址 qq邮箱是'smtp.qq.com', 163邮箱是'smtp.163.com'
        mail_server = 'smtp.163.com'
        now = time.strftime("%Y-%m-%d %H_%M_%S")
        # 邮件标题
        subject = '危险驾驶警告' + now
        # 发件人的邮箱及邮箱授权码
        username = send_email
        password = 'JJXTXHDXKPKBGGYT'  # 注意这里是邮箱的授权码而不是邮箱密码
        # 邮箱的内容和标题
        message = MIMEText(mail_body, 'html', 'utf8')
        message['Subject'] = Header(subject, charset='utf8')
        # 发送邮件，使用的使smtp协议
        smtp = smtplib.SMTP()
        smtp.connect(mail_server)
        smtp.login(username, password)
        smtp.sendmail(send_addr, reciver_addr.split(','), message.as_string())
        smtp.quit()


if __name__ == '__main__':
    # 自动发送邮件
    SendEmail().send_email('./report/message.txt')