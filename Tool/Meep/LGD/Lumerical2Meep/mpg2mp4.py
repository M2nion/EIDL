import subprocess

input_path = "/home/min/EIDL/Tool/Meep/LGD/Meep code/OLED_2D_movie_monitor_TE.mpg"
output_path = "/home/min/EIDL/Tool/Meep/LGD/Meep code/OLED_2D_movie_monitor_TE.mp4"

# ffmpeg 명령어 실행
subprocess.run([
    "ffmpeg", "-i", input_path,
    "-c:v", "libx264",
    "-crf", "23",
    "-preset", "veryfast",
    output_path
])
