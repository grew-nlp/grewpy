import time
import os, sys
import mmap

if __name__ == "__main__":
    filename = './channel_out.txt'
    fd = open(filename, "r+b")
    channel = mmap.mmap(fd.fileno(), 128, access=mmap.ACCESS_WRITE, offset=0)
    while True:
        pos = channel[0] + 1
        print(f"pos = {pos}")
        channel[0] = pos
        time.sleep(2)
    c_cout.close()
    
sys.exit(0)
