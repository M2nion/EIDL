import os
import time
from dotenv import load_dotenv  # pip install python-dotenv

from slack_sdk import WebClient  # pip install slack_sdk
from slack_sdk.errors import SlackApiError

load_dotenv()

token = os.getenv("BOT_TOKKEN")
channel = os.getenv("CHANNEL")

class Meepbot:
    def __init__(self, token: str = token, channel: str = channel):
        self.client = WebClient(token=token)
        self.default_channel = channel
        self.simulation_start_time = None  # 시뮬레이션 시작 시각

    def send_message(self, message: str, channel: str = None):
        target_channel = channel or self.default_channel
        try:
            self.client.chat_postMessage(channel=target_channel, text=message)
        except SlackApiError as e:
            print(f"Slack API 오류 (채널: {target_channel}): {e.response['error']}")
        except Exception as e:
            print(f"오류 발생 (채널: {target_channel}): {e}")

    def start_sim(self):
        self.simulation_start_time = time.time()
        print("시뮬레이션을 시작...")

    def end_sim(self):
        if self.simulation_start_time is not None:
            elapsed_time = time.time() - self.simulation_start_time
            # 메시지를 보냄
            self.send_message(f"시뮬레이션이 종료되었습니다.\n 코드 실행 시간: {elapsed_time:.2f}초")
            # 다음 시뮬레이션을 대비해 초기화
            self.simulation_start_time = None
        else:
            print("시뮬레이션 시작 시간이 기록되지 않았습니다.")

