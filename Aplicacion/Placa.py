# Esto es de Luis lo comento, para hacer pruebas.
# from flask import Flask, request
# from Modelos.Clasificacion_Sentimientos import clasificacion_sentimiento
# from Modelos.Traduccion import traduccion_texto

# app = Flask(__name__)

# @app.route('/procesar_texto', methods=['POST'])

# def recibir_texto():
#     if request.method == 'POST':
#         texto = request.form['texto1']
#         if 'submit_clasificacion' in request.form:
#             resultado = clasificacion_sentimiento(texto)
#         elif 'submit_traduccion' in request.form:
#             resultado = traduccion_texto(texto)
#         else:
#             resultado = 'Acción desconocida'
#         return resultado
  
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)


import socket
s = socket.socket()
s.bind(("192.168.88.219", 2223))
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

