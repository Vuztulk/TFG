from flask import Flask, request, jsonify
from Modelos.Clasificacion_Sentimientos.distilbert import sent_distilbert
from Modelos.Traductor.marian import trad_marian
from Modelos.Traductor.cartesinus import trad_cartenisus
from Modelos.Traductor.autotrain import trad_autotrain
from Modelos.Clasificacion_Sentimientos.roBERTa import sent_roberta
from Modelos.Clasificacion_Sentimientos.sbcBI import sent_sbcbi
from Modelos.Predictor_Texto.gpt2 import pred_gpt2
from Modelos.Resumen_Texto.bart import res_bart
from Modelos.Resumen_Texto.T5 import res_t5

app = Flask(__name__)

@app.route('/', methods=['POST'])

def recibir_texto():
    if request.method == 'POST':
        
        accion = request.form.get('accion')
        texto = request.form['texto']
        #modelo = request.form['modelo']
        modelo = 'cartenisus'
        
        funciones = {
            'clasificacion': {
                'distilbert': sent_distilbert,
                'roberta': sent_roberta,
                'sbcbi': sent_sbcbi
            },
            'traduccion': {
                'autotrain': trad_autotrain,
                'cartenisus': trad_cartenisus,
                'marian': trad_marian
            },
            'predictor': {
                'default': pred_gpt2
            },
            'resumen': {
                'bart': res_bart,
                't5': res_t5
            }
        }

        try:
            resultado, t_cpu, t_total = funciones[accion][modelo](texto)
        except KeyError:
            resultado = 'Acción desconocida'

        return jsonify({'resultado': resultado, 't_cpu': t_cpu, 't_total': t_total})
    
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=6000)




