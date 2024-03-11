from flask import Flask, request
from Modelos.Clasificacion_Sentimientos import distilbert, roberta, sbcbi
from Modelos.Traductor import trad_marian, trad_cartenisus, autotrain

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
            resultado = trad_marian(texto)
        #elif accion == 'predictor':
            #resultado = clasificacion_sentimiento(texto)
        #elif accion == 'resumen':
            #resultado = traduccion_texto(texto)
        else:
            resultado = 'Acción desconocida'
        return resultado
  
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=6000)




