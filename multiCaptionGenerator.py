import caption
import os
import subprocess


def main():
    subprocess.call(['python3', 'caption.py', '-i', 'VisualisationTest/flosexy.jpg', '-tvr', 'random'])
    subprocess.call(['python3', 'caption.py', '-i', 'VisualisationTest/scooter.jpg', '-tvr', 'validate'])
    subprocess.call(['python3', 'caption.py', '-i', 'VisualisationTest/biketour.jpg', '-tvr', 'train'])

if __name__ == '__main__':
    main()
