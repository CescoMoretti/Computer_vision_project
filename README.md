# Computer Vision Project

1-in .vscode folder create a file named tasks.json
2- paste this: 
    {
        "version": "2.0.0",
        "command": "Python",
        "isShellCommand": true,
        "args": [ "${file}",
            "--video_dir", "[path to test video].mp4",
            "--out_dir", "[path to a folder to save eventual test frames]"
        ],
        "showOutput": "always"
    }
3- fill the paths with your own
4- open the main.py file and press ctr+shift+b