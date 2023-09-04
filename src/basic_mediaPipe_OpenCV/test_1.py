import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm

import subprocess

in_vid = "./data_dir/00067cfb-caba8a02.mov"
print(in_vid)

subprocess.run(['ffmpeg' , '-i' ,in_vid])

