import os
from yt_dlp import YoutubeDL

def download_youtube_audio(url: str, output_dir: str = "downloads", quality: str = "192"):
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # yt-dlp 설정
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": quality,
        }],
        "quiet": True,      
        "noplaylist": True,  # 플레이리스트가 아닌 단일 동영상으로만 동작
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

# 예시: 사용자가 입력한 URL
video_url = input("다운로드할 유튜브 URL을 입력하세요: ").strip()
# 저장 경로 지정 (예: 컴퓨터 내 원하는 폴더, 또는 USB/휴대폰 마운트 경로)
save_folder = "/home/min/EIDL/Tool/Meep/Micro LED"

print(f"음원을 '{save_folder}' 폴더에 MP3로 저장합니다...")
download_youtube_audio(video_url, output_dir=save_folder)
print("다운로드가 완료되었습니다!")
