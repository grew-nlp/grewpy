import time
import os, sys
import mmap

channel_out_filename = './channel_out.txt'
channel_in_filename = 'channel_in.txt'

def get_channel(filename, is_w):
    fd = open(filename, "r+b")
    access = mmap.ACCESS_WRITE if is_w else mmap.ACCESS_READ
    channel = mmap.mmap(fd.fileno(), 128, access=access, offset=0)
    return fd, channel

if __name__ == "__main__":
    f_out, c_out = get_channel(channel_out_filename, True)
    f_in, c_in = get_channel(channel_in_filename, False)
    pos = 0
    while True:
        pos = pos + 1
        print(f"pos = {pos}")
        c_out.seek(0)
        c_out.write(pos.to_bytes(6, "little"))
        c_in.seek(0)
        x = c_in.read(1)
        if x == b'\x01':
            print("ici")
            pos = 0
        time.sleep(2)
    c_cout.close()
    
sys.exit(0)