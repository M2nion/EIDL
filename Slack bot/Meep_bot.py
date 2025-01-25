import os
from dotenv import load_dotenv

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

load_dotenv()

token = os.getenv("BOT_TOKKEN")
channel = os.getenv("CHANNEL")

class Meepbot:
    def __init__(self, token: str = token, channel: str = channel):
        self.client = WebClient(token=token)
        self.default_channel = channel

    def send_message(self, message: str, channel: str = None):
        target_channel = channel or self.default_channel
        try:
            self.client.chat_postMessage(channel=target_channel, text=message)
            print("메시지 전송 성공")
        except SlackApiError as e:
            print(f"Slack API 오류 (채널: {target_channel}): {e.response['error']}")
        except Exception as e:
            print(f"오류 발생 (채널: {target_channel}): {e}")
