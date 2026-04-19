import sys
import os
import subprocess
import threading
import time
import webbrowser

def get_app_path():
    """Get the path to pj7.py whether running as script or frozen exe."""
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, 'pj7.py')
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pj7.py')

def open_browser():
    """Open browser after a short delay to let Streamlit start."""
    time.sleep(3)
    webbrowser.open('http://localhost:8501')

def main():
    app_path = get_app_path()
    threading.Thread(target=open_browser, daemon=True).start()

    if getattr(sys, 'frozen', False):
        from streamlit.web import cli as stcli
        sys.argv = ['streamlit', 'run', app_path,
                     '--global.developmentMode=false',
                     '--server.headless=true',
                     '--server.port=8501',
                     '--browser.gatherUsageStats=false']
        stcli.main()
    else:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', app_path,
            '--server.headless=true',
            '--server.port=8501',
            '--browser.gatherUsageStats=false'
        ])

if __name__ == '__main__':
    main()
