import socket

s = socket.socket()

try:
    s.connect(("192.168.88.219", 22))
    s.send("a".encode())
    print("Datos enviados correctamente")
except Exception as e:
    print("Error al enviar datos:", e)
finally:
    s.close()