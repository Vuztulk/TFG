from flask import Flask, render_template, request
import requests

app = Flask(__name__)

def procesar_solicitud(placa, texto, modelo):
    if placa == 'local':
        url = 'http://127.0.0.1:6000'
    elif placa == 'rasperri':
        url = 'URL_para_rasperri'
    elif placa == 'orin-cpu':
        url = 'URL_para_orin-cpu'
    elif placa == 'orin-gpu':
        url = 'URL_para_orin-gpu'
    else:
        return "Placa no reconocida"
    
    response = requests.post(url, data={'accion': 'traduccion', 'texto': texto, 'modelo': modelo})
    return response.text

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/traductor', methods=['GET', 'POST'])
def traductor():
    if request.method == 'POST':
        texto = request.form.get('texto')
        placa = request.form.get('placa')
        modelo = request.form.get('modelo')
        resultado = procesar_solicitud(placa, texto, modelo)
        return render_template('traductor.html', resultado=resultado)
    
    return render_template('traductor.html')

@app.route('/clasificacion_sentimientos', methods=['GET', 'POST'])
def clasificacion_sentimientos():
    if request.method == 'POST':
        texto = request.form.get('texto')
        placa = request.form.get('placa')
        modelo = request.form.get('modelo')
        
        resultado = procesar_solicitud(placa, texto, modelo)
        return render_template('clasificacion_sentimientos.html', resultado=resultado)
    
    return render_template('clasificacion_sentimientos.html')

@app.route('/predictor_texto', methods=['GET', 'POST'])
def predictor_texto():
    if request.method == 'POST':
        texto = request.form.get('texto')
        placa = request.form.get('placa')
        modelo = request.form.get('modelo')
        
        resultado = procesar_solicitud(placa, texto, modelo)
        return render_template('predictor_texto.html', resultado=resultado)
    
    return render_template('predictor_texto.html')

@app.route('/resumen_texto', methods=['GET', 'POST'])
def resumen_texto():
    if request.method == 'POST':
        texto = request.form.get('texto')
        placa = request.form.get('placa')
        modelo = request.form.get('modelo')
        
        resultado = procesar_solicitud(placa, texto, modelo)
        return render_template('resumen_texto.html', resultado=resultado)
    
    return render_template('resumen_texto.html')

if __name__ == '__main__':
    app.run()
