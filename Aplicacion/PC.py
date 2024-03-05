# Esto es de Luis lo comento, para hacer pruebas.
# from flask import Flask, render_template, request
# import requests

# app = Flask(__name__)

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         texto = request.form.get('texto')
#         response = requests.post('http://other-ip/process', data={'text': texto})
#         processed_text = response.text
#         return render_template('index.html', processed_text=processed_text)
#     return render_template('index.html')

# if __name__ == '__main__':
#     app.run(debug=True)

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
