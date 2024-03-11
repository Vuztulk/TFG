from flask import Flask, request
from Modelos.Clasificacion_Sentimientos.distilbert import sent_distilbert
from Modelos.Traductor.marian import trad_marian

app = Flask(__name__)

@app.route('/', methods=['POST'])

def recibir_texto():
    if request.method == 'POST':
        texto = request.form['texto']
        placa = request.form['placa']

        accion = request.form.get('accion')
        if accion == 'clasificacion':
            resultado, t_cpu, t_total = sent_distilbert(texto)
        elif accion == 'traduccion':
            resultado, t_cpu, t_total = trad_marian(texto)
        #elif accion == 'predictor':
            #resultado, t_cpu, t_total = clasificacion_sentimiento(texto)
        #elif accion == 'resumen':
            #resultado, t_cpu, t_total = traduccion_texto(texto)
        else:
            resultado = 'Acción desconocida'
        return resultado
  
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=6000)




