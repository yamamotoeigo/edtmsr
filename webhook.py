import requests

# webhookを行う関数
def send_webhook(name):
    data = {
        "content": f"{name}が完了したよ"
    }
    url = "https://discord.com/api/webhooks/1263421613450461275/Srq2HBx7n8_Tvlyu87zAGwDKNyTHqqyXAQjloMypHytYJN8eoqM_03Zw78G2xkKjxIXE"
    requests.post(url, data=data)