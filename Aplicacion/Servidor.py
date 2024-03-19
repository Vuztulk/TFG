from flask import Flask, request, jsonify
import torch
import os
import argparse
from Modelos.Clasificacion_Sentimientos.distilbert import sent_distilbert_cpu, sent_distilbert_gpu
from Modelos.Traductor.marian import trad_marian_cpu, trad_marian_gpu
from Modelos.Traductor.cartesinus import trad_cartenisus_cpu, trad_cartenisus_gpu
from Modelos.Traductor.autotrain import trad_autotrain_cpu, trad_autotrain_gpu
from Modelos.Clasificacion_Sentimientos.roBERTa import sent_roberta_cpu, sent_roberta_gpu
from Modelos.Clasificacion_Sentimientos.sbcBI import sent_sbcbi_cpu, sent_sbcbi_gpu
from Modelos.Predictor_Texto.gpt2 import pred_gpt2_cpu, pred_gpt2_gpu
from Modelos.Resumen_Texto.bart import res_bart_cpu, res_bart_gpu
from Modelos.Resumen_Texto.T5 import res_t5_cpu, res_t5_gpu
from Modelos.Conexion import comprobar_conexion

app = Flask(__name__)

@app.route('/', methods=['POST'])
def recibir_texto():
    if request.method == 'POST':
        
        accion = request.form['accion']
        texto = request.form['texto']
        modelo = request.form['modelo']
        procesador = request.form['procesador']
        longitud = int(request.form['longitud'])
        
        funciones = {
            'clasificacion': {
                'distilbert': (sent_distilbert_cpu, sent_distilbert_gpu),
                'roberta': (sent_roberta_cpu, sent_roberta_gpu),
                'sbcbi': (sent_sbcbi_cpu, sent_sbcbi_gpu)
            },
            'traduccion': {
                'autotrain': (trad_autotrain_cpu, trad_autotrain_gpu),
                'cartenisus': (trad_cartenisus_cpu, trad_cartenisus_gpu),
                'marian': (trad_marian_cpu, trad_marian_gpu)
            },
            'predictor': {
                'gpt2': (pred_gpt2_cpu, pred_gpt2_gpu)
            },
            'resumen': {
                'bart': (res_bart_cpu, res_bart_gpu),
                't5': (res_t5_cpu, res_t5_gpu)
            },
            'conexion': {
                'conexion': (comprobar_conexion, comprobar_conexion),
            }
        }

        try:
            cpu_func, gpu_func = funciones[accion][modelo]
            if procesador == 'gpu' and torch.cuda.is_available():
                resultado, t_cpu, t_total, memoria = gpu_func(texto, longitud)
            else:
                resultado, t_cpu, t_total, memoria = cpu_func(texto, longitud)
        except KeyError:
            resultado = 'Acción desconocida'

        return jsonify({'resultado': resultado, 't_cpu': t_cpu, 't_total': t_total, 'memoria': memoria})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', action='store_true', help='Argumento "r"')
    parser.add_argument('-o', action='store_true', help='Argumento "o"')
    parser.add_argument('-l', action='store_true', help='Argumento "l"')
    parser.add_argument('-p', type=int, default=8888, help='Puerto para ejecutar el servidor (default: 8888)')

    args = parser.parse_args()

    host = '127.0.0.1'

    if args.r:
        host = ''
    elif args.o:
        host = '0.0.0.0'
    elif args.l:
        host = '127.0.0.1'
        
    app.run(host=host, port=args.p)

