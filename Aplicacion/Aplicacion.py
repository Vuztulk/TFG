from flask import Flask, render_template, request
import requests, ping3

app = Flask(__name__)

URLS = {
    'Local': 'http://127.0.0.1:8888',
    'Raspberry': 'http://sardina.dacya.ucm.es:16992',
    'Orin-CPU': 'http://sardina.dacya.ucm.es:16992',
    'Orin-GPU': 'http://sardina.dacya.ucm.es:16992'
}

def procesar_solicitud(accion, placa, texto, modelo, longitud, procesador='cpu'):
    url = URLS.get(placa)
    if not url:
        return "Placa no reconocida"
    if placa == 'Orin-GPU':
        procesador = 'gpu'
    response = requests.post(url, data={'accion': accion, 'texto': texto, 'modelo': modelo, 'longitud': longitud, 'procesador': procesador})
    return response.json()

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/conexion', methods=['GET', 'POST'])
def conexion():
    if request.method == 'POST':
        placa = request.form.get('placa')
        url = URLS.get(placa)
        if url:
            host = url.split('//')[1].split(':')[0]  # Extraer el host de la URL
            resultado_ping = ping3.ping(host)

            if resultado_ping == 'None':
                estado = "OFF"
            else:
                estado = "ON"
        else:
            estado = "Placa no reconocida"
       
        return render_template('conexion.html', estado=estado, placa=placa)
    return render_template('conexion.html')

@app.route('/<ruta>', methods=['GET', 'POST'])
def ruta(ruta):
    if request.method == 'POST':
        
        texto = request.form.get('texto')
        placa = request.form.get('placa')
        modelo = request.form.get('modelo')
        longitud = request.form.get('longitud', 0)
        
        accion = 'traduccion' if ruta == 'traductor' else 'clasificacion' if ruta == 'clasificacion_sentimientos' else 'predictor' if ruta == 'predictor_texto' else 'resumen'
        response_data = procesar_solicitud(accion, placa, texto, modelo, longitud)
        
        resultado = response_data.get('resultado', '')
        t_cpu = response_data.get('t_cpu', '')
        t_total = response_data.get('t_total', '')
        memoria = response_data.get('memoria', '')
        
        if memoria < 0.000:
            memoria = "Memoria utilizada demasiado baja"
            
        return render_template(f'{ruta}.html', resultado=resultado, placa=placa, t_cpu=t_cpu, t_total=t_total, modelo=modelo, longitud=longitud, memoria=memoria)
    
    return render_template(f'{ruta}.html')

if __name__ == '__main__':
    app.run()
