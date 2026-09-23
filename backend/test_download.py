import yt_dlp

FFMPEG_PATH = r"D:\Projects\TruthCheck\backend\venv\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe"

def download_video(url, output_path="test_video.mp4"):
    ydl_opts = {
        "outtmpl": output_path,
        "format": "bestvideo[vcodec^=avc1][height<=480]+bestaudio/best[height<=480]",
        "ffmpeg_location": FFMPEG_PATH
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

if __name__ == "__main__":
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    download_video(test_url)
    print("Download complete")