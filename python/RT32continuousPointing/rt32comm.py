'''
Created on Feb 2, 2022

@author: blew
'''
import os,sys
import socket
import time

class rt32tcpclient():
    def __init__(self):
        '''
        '''
        self.RT4sock=None
        if sys.version_info >= (2, 7):
            self.RT4sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        else:
            self.RT4sock = socket.socket()

    def connectRT4(self,host='192.168.1.4',port=3490):
        
        
        # Connect the socket to the port where the server is listening
        server_address = (host, int(port))
    #     print >>sys.stderr, 'connecting to %s port %s' % server_address
        if sys.version_info >= (2, 7):
            self.RT4sock = socket.create_connection(server_address)
        else:
            self.RT4sock.connect(server_address)
    
        cmd='REC\n'
        self.RT4sock.sendall(cmd.encode())
    #     time.sleep(0.2)
    
    
        return self


    def send_cmd(self,cmd):
        # data = self.RT4sock.recv(16000)
        cmd+='\0'
        self.RT4sock.sendall(cmd.encode())
        time.sleep(0.5)
    #     RT4sock.close()
        # data = self.RT4sock.recv(16000)
        data=None
        return data #.decode()
    


def getCurrentContinuousCorrections(host='192.168.1.4',port=33033,ans_host='192.168.1.255',ans_port=33036):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        DATA='7'
        s.bind((ans_host, ans_port))
        s.sendto(DATA.encode(), (host, int(port)))
        
        data, addr = s.recvfrom(1500)
        dstr=data.decode()
        dstr=dstr[:-1] if dstr.endswith(',') else dstr
        
        pointing_stats={}
        for kv in dstr.split(','):
            k,v=kv.split('=')
            pointing_stats[k]=v.strip()
    except OSError:
        print('cound not get current continuous corrections from RT32')
        return None
    return pointing_stats


