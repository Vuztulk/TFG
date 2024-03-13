from flask import Flask, render_template, request
import requests

app = Flask(__name__)

def procesar_solicitud(accion, placa, texto, modelo, procesador = 'cpu'):
    
    if placa == 'Local':
        url = 'http://127.0.0.1:6000'
    elif placa == 'Raspberry':
        url = 'URL_para_rasperri'
    elif placa == 'Orin-CPU':
        url = 'URL_para_orin-cpu'
    elif placa == 'Orin-GPU':
        url = 'URL_para_orin-gpu'
        procesador = 'gpu'
    else:
        return "Placa no reconocida"
    
    return requests.post(url, data={'accion': accion, 'texto': texto, 'modelo': modelo, 'procesador': procesador}).json().values()

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/traductor', methods=['GET', 'POST'])
def traductor():
    if request.method == 'POST':
        texto = request.form.get('texto')
        placa = request.form.get('placa')
        modelo = request.form.get('modelo')
        resultado, t_cpu, t_total = procesar_solicitud('traduccion', placa, texto, modelo)
        return render_template('traductor.html', resultado=resultado, placa=placa)
    
    return render_template('traductor.html')

@app.route('/clasificacion_sentimientos', methods=['GET', 'POST'])
def clasificacion_sentimientos():
    if request.method == 'POST':
        texto = request.form.get('texto')
        placa = request.form.get('placa')
        modelo = request.form.get('modelo')
        resultado, t_cpu, t_total = procesar_solicitud('clasificacion', placa, texto, modelo)
        return render_template('clasificacion_sentimientos.html', resultado=resultado)
    
    return render_template('clasificacion_sentimientos.html')

@app.route('/predictor_texto', methods=['GET', 'POST'])
def predictor_texto():
    if request.method == 'POST':
        texto = request.form.get('texto')
        placa = request.form.get('placa')
        modelo = request.form.get('modelo')
        resultado, t_cpu, t_total = procesar_solicitud('predictor', placa, texto, modelo)
        return render_template('predictor_texto.html', resultado=resultado)
    
    return render_template('predictor_texto.html')

@app.route('/resumen_texto', methods=['GET', 'POST'])
def resumen_texto():
    if request.method == 'POST':
        texto = request.form.get('texto')
        placa = request.form.get('placa')
        modelo = request.form.get('modelo')
        resultado, t_cpu, t_total = procesar_solicitud('resumen', placa, texto, modelo)
        return render_template('resumen_texto.html', resultado=resultado)
    
    return render_template('resumen_texto.html')

if __name__ == '__main__':
    app.run()
