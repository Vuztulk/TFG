
from flask import Flask, request
from Modelos.Clasificacion_Sentimientos import clasificacion_sentimiento
from Modelos.Traduccion import traduccion_texto

app = Flask(__name__)

@app.route('/', methods=['POST'])

def recibir_texto():
    if request.method == 'POST':
        texto = request.form['texto']
        placa = request.form['placa']
        print(placa)
        accion = request.form.get('accion')
        if accion == 'clasificacion':
            resultado = clasificacion_sentimiento(texto)
        elif accion == 'traduccion':
            resultado = traduccion_texto(texto)
        elif accion == 'predictor':
            resultado = clasificacion_sentimiento(texto)
        elif accion == 'resumen':
            resultado = traduccion_texto(texto)
        else:
            resultado = 'Acción desconocida'
        return resultado
  
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=6000)




