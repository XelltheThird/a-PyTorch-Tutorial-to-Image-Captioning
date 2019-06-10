import matplotlib as mpl
mpl.use('Pdf')

import argparse
import caption
import os
import subprocess


def main(file_path):
	if not os.path.isdir(file_path):
		os.mkdir(file_path)
	subprocess.call(['python3', 'caption.py', '-i', 'VisualisationTest/flosexy.jpg', '-tvr', 'random', '-f', file_path])
	subprocess.call(['python3', 'caption.py', '-i', 'VisualisationTest/scooter.jpg', '-tvr', 'validate', '-f', file_path])
	subprocess.call(['python3', 'caption.py', '-i', 'VisualisationTest/biketour.jpg', '-tvr', 'train', '-f', file_path])

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='MultiCaptionGenerator')

	parser.add_argument('--folder', '-f', default="testing/", help='folder')
	args = parser.parse_args()
	main(args.folder)