from flask import Flask, render_template, request
import requests

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/traductor', methods=['GET', 'POST'])
def traductor():
    if request.method == 'POST':
        texto = request.form.get('texto')
        placa = request.form.get('placa')
        response = requests.post('http://127.0.0.1:6000', data={'accion': 'traduccion', 'texto': texto, 'placa': placa}) #Quedaria por añadir el modelo con el cual se quiere procesar
        processed_text = response.text
        return render_template('traductor.html', resultado=processed_text)
    return render_template('traductor.html')

@app.route('/clasificacion_sentimientos', methods=['GET', 'POST'])
def sentimientos():
    if request.method == 'POST':
        texto = request.form.get('texto')
        placa = request.form.get('placa')
        response = requests.post('http://127.0.0.1:6000', data={'accion': 'clasificacion', 'texto': texto, 'placa': placa})
        processed_text = response.text
        return render_template('clasificacion_sentimientos.html', resultado=processed_text)
    return render_template('clasificacion_sentimientos.html')

@app.route('/predictor_texto', methods=['GET', 'POST'])
def predictor():
    if request.method == 'POST':
        texto = request.form.get('texto')
        placa = request.form.get('placa')
        response = requests.post('http://127.0.0.1:6000', data={'accion': 'predictor', 'texto': texto, 'placa': placa})
        processed_text = response.text
        return render_template('predictor_texto.html', resultado=processed_text)
    return render_template('predictor_texto.html')

@app.route('/resumen_texto', methods=['GET', 'POST'])
def resumen():
    if request.method == 'POST':
        texto = request.form.get('texto')
        placa = request.form.get('placa')
        response = requests.post('http://127.0.0.1:6000', data={'accion': 'resumen', 'texto': texto, 'placa': placa})
        processed_text = response.text
        return render_template('resumen_texto.html', resultado=processed_text)
    return render_template('resumen_texto.html')

if __name__ == '__main__':
    app.run()
