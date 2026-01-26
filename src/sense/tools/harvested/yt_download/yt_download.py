import sys
import yt_dlp

def download_video(url):
    print(f"[yt_download] Processing: {url}")
    
    # Configure options for resilience
    ydl_opts = {
        'format': 'best',
        'noplaylist': True,
        'quiet': True,
        'no_warnings': True,
        # We don't need 'outtmpl' paths because the Universal Adapter 
        # already forces the CWD to /sdcard/Download
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get('title', 'Unknown Title')
            return f"Success: Downloaded '{title}'"
            
    except Exception as e:
        return f"Download Error: {str(e)}"

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # The Universal Adapter passes the URL as the first argument
        print(download_video(sys.argv[1]))
    else:
        print("Usage: python yt_download.py [url]")
