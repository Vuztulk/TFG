import socket
s = socket.socket()
s.bind(("192.168.88.219", 22))
s.listen(10)

while True:
    (sc, addrc) = s.accept()
    continuar = True
    while continuar:
        dato = sc.recv(64)
        if not dato:
            continuar = False
            print("Cliente desconectado")
        else:
            dato2 = dato.decode()
            if dato2 == "a":
                print("Dato a")

s.close()
print("Fin del programa")

