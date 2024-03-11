
from flask import Flask, request
from Modelos.Clasificacion_Sentimientos.distilbert import distilbert
from Modelos.Clasificacion_Sentimientos.roBERTa import 
from Modelos.Clasificacion_Sentimientos.sbcBI import distilbert
from Modelos.Traductor.marian_es_en import marian
from Modelos.Traductor. import marian
from Modelos.Traductor.marian_es_en import marian

app = Flask(__name__)

@app.route('/', methods=['POST'])

def recibir_texto():
    if request.method == 'POST':
        texto = request.form['texto']
        placa = request.form['placa']

        accion = request.form.get('accion')
        if accion == 'clasificacion':
            resultado = distilbert(texto)
        elif accion == 'traduccion':
            resultado = marian(texto)
        elif accion == 'predictor':
            resultado = clasificacion_sentimiento(texto)
        elif accion == 'resumen':
            resultado = traduccion_texto(texto)
        else:
            resultado = 'Acción desconocida'
        return resultado
  
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=6000)




