
from flask import Flask, request
from Modelos.Clasificacion_Sentimientos import clasificacion_sentimiento
from Modelos.Traduccion import traduccion_texto

app = Flask(__name__)

@app.route('/procesar_texto', methods=['POST'])

def recibir_texto():
    if request.method == 'POST':
        texto = request.form['texto']
        if 'submit_clasificacion' in request.form:
            resultado = clasificacion_sentimiento(texto)
        elif 'submit_traduccion' in request.form:
            resultado = traduccion_texto(texto)
        else:
            resultado = 'Acción desconocida'
        return resultado
  
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)




