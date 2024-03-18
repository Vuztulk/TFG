from flask import Flask, render_template, request
import requests

app = Flask(__name__)

def procesar_solicitud(accion, placa, texto, modelo, longitud, procesador = 'cpu'):
    
    if placa == 'Local':
        url = 'http://127.0.0.1:8888'
    elif placa == 'Raspberry':
        url = 'http://sardina.dacya.ucm.es:16992'
    elif placa == 'Orin-CPU':
        url = 'http://sardina.dacya.ucm.es:16992'
    elif placa == 'Orin-GPU':
        url = 'http://sardina.dacya.ucm.es:16992'
        procesador = 'gpu'
    else:
        return "Placa no reconocida"
    
    response = requests.post(url, data={'accion': accion, 'texto': texto, 'modelo': modelo, 'longitud': longitud, 'procesador': procesador})
    return response.json()

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/traductor', methods=['GET', 'POST'])
def traductor():
    if request.method == 'POST':
        texto = request.form.get('texto')
        placa = request.form.get('placa')
        modelo = request.form.get('modelo')
        response_data = procesar_solicitud('traduccion', placa, texto, modelo, 0)
        resultado = response_data.get('resultado', '')
        t_cpu = response_data.get('t_cpu', '')
        t_total = response_data.get('t_total', '')
        memoria = response_data.get('memoria', '')

        if memoria < 0.000:
            memoria = "Memoria utilizada demasiado baja"

        return render_template('traductor.html', resultado=resultado, placa=placa, t_cpu=t_cpu, t_total=t_total, modelo=modelo, memoria=memoria)
    
    return render_template('traductor.html')

@app.route('/clasificacion_sentimientos', methods=['GET', 'POST'])
def clasificacion_sentimientos():
    if request.method == 'POST':
        texto = request.form.get('texto')
        placa = request.form.get('placa')
        modelo = request.form.get('modelo')
        response_data = procesar_solicitud('clasificacion', placa, texto, modelo, 0)
        resultado = response_data.get('resultado', '')
        t_cpu = response_data.get('t_cpu', '')
        t_total = response_data.get('t_total', '')
        memoria = response_data.get('memoria', '')

        if memoria < 0.000:
            memoria = "Memoria utilizada demasiado baja"

        return render_template('clasificacion_sentimientos.html', resultado=resultado, placa=placa, t_cpu=t_cpu, t_total=t_total, modelo=modelo, memoria=memoria)
    
    return render_template('clasificacion_sentimientos.html')

@app.route('/predictor_texto', methods=['GET', 'POST'])
def predictor_texto():
    if request.method == 'POST':
        texto = request.form.get('texto')
        placa = request.form.get('placa')
        modelo = request.form.get('modelo')
        longitud = request.form.get('longitud')
        response_data = procesar_solicitud('predictor', placa, texto, modelo, longitud)
        resultado = response_data.get('resultado', '')
        t_cpu = response_data.get('t_cpu', '')
        t_total = response_data.get('t_total', '')
        memoria = response_data.get('memoria', '')

        if memoria < 0.000:
            memoria = "Memoria utilizada demasiado baja"

        return render_template('predictor_texto.html', resultado=resultado, placa=placa, t_cpu=t_cpu, t_total=t_total, modelo=modelo, longitud=longitud, memoria=memoria)
    
    return render_template('predictor_texto.html')

@app.route('/resumen_texto', methods=['GET', 'POST'])
def resumen_texto():
    if request.method == 'POST':
        texto = request.form.get('texto')
        placa = request.form.get('placa')
        modelo = request.form.get('modelo')
        longitud = request.form.get('longitud')
        response_data = procesar_solicitud('resumen', placa, texto, modelo, longitud)
        resultado = response_data.get('resultado', '')
        t_cpu = response_data.get('t_cpu', '')
        t_total = response_data.get('t_total', '')
        memoria = response_data.get('memoria', '')

        if memoria < 0.000:
            memoria = "Memoria utilizada demasiado baja"

        return render_template('resumen_texto.html', resultado=resultado, placa=placa, t_cpu=t_cpu, t_total=t_total, modelo=modelo, longitud=longitud, memoria=memoria)
    
    return render_template('resumen_texto.html')

if __name__ == '__main__':
    app.run()
