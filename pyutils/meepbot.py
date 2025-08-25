# meepbot.py
import os, time, socket, inspect, sys
from typing import Optional

# ─────────────────────────────────────────────────────────────────────
# 1) .env 자동 로드: 가장 가까운 .env → 없으면 ~/.env
#    (python-dotenv 미설치/오류 시 조용히 패스)
# ─────────────────────────────────────────────────────────────────────
def _auto_load_env():
    try:
        from dotenv import load_dotenv, find_dotenv
        # 현재 작업 디렉터리부터 상위로 .env 탐색
        path = find_dotenv(usecwd=True)
        loaded = False
        if path:
            loaded = load_dotenv(path, override=False)
        if not loaded:
            # 홈 디렉터리의 .env도 시도
            home_env = os.path.expanduser("~/.env")
            if os.path.exists(home_env):
                load_dotenv(home_env, override=False)
    except Exception:
        pass

_auto_load_env()

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

def _format_hms(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def _infer_script_filename() -> str:
    """
    실행한 스크립트 파일명(가능하면 __main__의 __file__)을 반환.
    실패 시 sys.argv[0] 또는 'python'으로 폴백.
    """
    try:
        # __main__ 프레임에서 __file__을 찾는 것이 가장 정확
        for fr in inspect.stack():
            glb = fr.frame.f_globals
            if glb.get("__name__") == "__main__" and "__file__" in glb:
                return os.path.basename(glb["__file__"])
        # 폴백: 가장 바깥쪽 프레임 파일
        return os.path.basename(inspect.stack()[-1].filename)
    except Exception:
        return os.path.basename(sys.argv[0]) or "python"

class MeepNotifier:
    """
    사용법 1) 컨텍스트 매니저
        with MeepNotifier(send_start=True, job="옵션"):
            sim.run(until=...)

    사용법 2) 데코레이터
        @notify_on_finish(send_start=True, job="옵션")
        def main(): ...
    """
    def __init__(
        self,
        token: Optional[str] = None,
        channel: Optional[str] = None,
        send_start: bool = True,
        job: Optional[str] = None,   # ← 네가 주면 그걸 쓰고, 아니면 파일명 자동 사용
    ):
        # 토큰
        token = token or os.getenv("SLACK_BOT_TOKEN") or os.getenv("BOT_TOKEN") or os.getenv("BOT_TOKKEN")
        if not token:
            raise RuntimeError("Slack token이 없습니다. SLACK_BOT_TOKEN(또는 BOT_TOKEN)을 설정하세요.")
        self.client = WebClient(token=token)

        # 채널
        ch = channel or os.getenv("SLACK_CHANNEL_ID") or os.getenv("SLACK_CHANNEL") or os.getenv("CHANNEL")
        if not ch:
            raise RuntimeError("Slack 채널이 없습니다. SLACK_CHANNEL_ID(또는 SLACK_CHANNEL/#name)를 설정하세요.")
        self.channel = self._resolve_channel(ch)

        self.send_start = send_start
        # 우선순위: 인자 job > 환경변수 JOB_NAME > 실행 파일명
        self.job  = job or os.getenv("JOB_NAME") or _infer_script_filename()
        self.host = socket.gethostname()
        self.t0   = None

    def _resolve_channel(self, ch: str) -> str:
        if ch[:1] in {"C", "G", "D"} and len(ch) >= 9:
            return ch
        # 채널 이름을 받은 경우 ID로 변환 시도(권한 필요: conversations:read)
        name = ch[1:] if ch.startswith("#") else ch
        try:
            cursor = None
            while True:
                resp = self.client.conversations_list(
                    limit=1000, cursor=cursor, types="public_channel,private_channel"
                )
                for c in resp.get("channels", []):
                    if c.get("name") == name:
                        return c["id"]
                cursor = resp.get("response_metadata", {}).get("next_cursor") or None
                if not cursor:
                    break
        except SlackApiError as e:
            print(f"[meepbot] 채널명→ID 변환 실패('{name}'): {e.response.get('error')}")
        return ch  # 실패 시 입력값 그대로

    def send(self, text: str):
        try:
            self.client.chat_postMessage(channel=self.channel, text=text)
        except SlackApiError as e:
            print(f"[meepbot] Slack API 오류: {e.response.get('error')}")
        except Exception as e:
            print(f"[meepbot] 전송 오류: {e}")

    def __enter__(self):
        self.t0 = time.time()
        if self.send_start:
            self.send(f"🚀 {self.job} started\nhost: {self.host}")
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = _format_hms(time.time() - self.t0) if self.t0 else "unknown"
        if exc:
            self.send(f"❌ {self.job} failed on {self.host}\nreason: {exc}\nelapsed: {elapsed}")
            return False
        else:
            self.send(f"✅ {self.job} finished on {self.host}\nelapsed: {elapsed}")
            return False

def notify_on_finish(**opts):
    def _decorator(fn):
        def _wrap(*args, **kwargs):
            with MeepNotifier(**opts):
                return fn(*args, **kwargs)
        return _wrap
    return _decorator
