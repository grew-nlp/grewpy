''' Utility tools to connect to ocaml GREW'''

from subprocess import Popen, PIPE
import time
import socket
import os.path
import json
import os
import sys

from .grew import GrewError

host = 'localhost'
port = 8888
remote_ip = ''
caml_pid = None

request_counter = 0 #number of request to caml

import signal
def preexec_function ():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def pid_exist(pid):
    try:
        os.kill(pid,0)
        return 1
    except:
        return 0

def init():
    global port, remote_ip, caml_pid
    grewpy = "grewpy_backend"
    if not pid_exist(caml_pid):
        python_pid = os.getpid()
        caml = Popen(
            [grewpy, "--caller", str(python_pid)],
            preexec_fn=preexec_function,
            stdout=PIPE
        )
        port = int(caml.stdout.readline().strip())
        caml_pid = caml.pid
        #wait for grew's lib answer
        time.sleep(0.1)
        if caml.poll() == None:
            print ("connected to port: " + str(port), file=sys.stderr)
            remote_ip = socket.gethostbyname(host)
            return (caml)
        else:
            print ("Failed to connect", file=sys.stderr)
            exit (1)

def connect():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((remote_ip, port))
        return s
    except socket.error:
        raise GrewError('Failed to create socket. Make sure that you have called grew.init.')
    except socket.gaierror:
        print('[GREW] Hostname could not be resolved. Sorry\n', file=sys.stderr)


packet_size=32768

def send_and_receive(msg):
    global request_counter
    try:
        request_counter += 1
        stocaml = connect()
        json_msg = json.dumps(msg).encode(encoding='UTF-8')
        len_string = "%010d" % len(json_msg)
        stocaml.sendall(len_string.encode(encoding='UTF-8'))

        packet_nb = len(json_msg) // packet_size
        for i in range (packet_nb):
            stocaml.sendall(json_msg[packet_size*i:packet_size*(i+1)])
        stocaml.sendall(json_msg[packet_nb*packet_size:])
        camltos = bytes()
        reply_len = int(stocaml.recv(10))

        camltos = b''
        while len(camltos) < reply_len:
            packet = stocaml.recv(reply_len - len(camltos))
            if not packet:
                return None
            camltos += packet
        stocaml.close()

        reply = json.loads(camltos.decode(encoding='UTF-8'))
        if reply["status"] == "OK":
            try:
                return reply["data"]
            except:
                return None
        elif reply["status"] == "ERROR":
            raise GrewError({"function": msg["command"], "message": reply["message"]})
    except socket.error:
        raise GrewError({"function": msg["command"], "message" : 'Socket error'})
    except AttributeError as e: # connect issue
        raise GrewError({"function": msg["command"], "message" : e.value})

