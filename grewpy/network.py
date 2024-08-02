''' Utility tools to connect to ocaml GREW'''

from subprocess import Popen, PIPE
import time
import socket
import os.path
import json, re
import os
import sys

from .grew import GrewError

host = 'localhost'
port = None
remote_ip = ''
caml_pid = None
minimal_grewpy_backend_version = "0.5.4"

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
    if pid_exist(caml_pid):
        print ("grewpy_backend already started", file=sys.stderr)
    else:
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
            check_version()
            print ("connected to port: " + str(port), file=sys.stderr)
            remote_ip = socket.gethostbyname(host)
            return (caml)
        else:
            print ("Failed to connect", file=sys.stderr)
            exit (1)

def connect():
    global caml_pid
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((remote_ip, port))
        return s
    except socket.error:
        caml_pid = None
        raise GrewError('Failed to create socket. grewpy_backend seems down. Run grew.init() to restart.')


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

# Source: https://www.tutorialspoint.com/compare-version-numbers-in-python
def compareVersion(version1, version2):
   versions1 = [int(v) for v in version1.split(".")]
   versions2 = [int(v) for v in version2.split(".")]
   for i in range(max(len(versions1),len(versions2))):
      v1 = versions1[i] if i < len(versions1) else 0
      v2 = versions2[i] if i < len(versions2) else 0
      if v1 > v2:
         return 1
      elif v1 < v2:
         return -1
   return 0

def check_version():
    req = { "command": "get_version" }
    current_version = send_and_receive(req)
    current_version = re.match("[^-]*", current_version).group(0)
    if compareVersion (current_version, minimal_grewpy_backend_version) < 0:
        print (f"Incompatible grewpy_backend version.", file=sys.stderr)
        print (f"You have version {current_version}, but it should be {minimal_grewpy_backend_version} or higher", file=sys.stderr)
        print (f"Please upgrade grewpy_backend (see https://grew.fr/usage/python#upgrade)", file=sys.stderr)

def check_be_version():
    req = { "command": "get_version" }
    current_version = send_and_receive(req)
    if compareVersion (current_version, minimal_grewpy_backend_version) < 0:
        print (f"Incompatible grewpy_backend version.", file=sys.stderr)
        print (f"You have version {current_version}, but it should be {minimal_grewpy_backend_version} or higher", file=sys.stderr)
        print (f"Please upgrade grewpy_backend (see https://grew.fr/usage/python#upgrade)", file=sys.stderr)
