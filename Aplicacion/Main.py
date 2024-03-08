from flask import Flask, render_template, request
from Modelos.Clasificacion_Sentimientos import clasificacion_sentimiento
from Modelos.Traduccion import traduccion_texto

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/traductor', methods=['GET', 'POST'])
def traductor():
    resultado = None
    if request.method == 'POST':
        texto = request.form['texto']
        if 'submit_clasificacion' in request.form:
            resultado = clasificacion_sentimiento(texto)
        elif 'submit_traduccion' in request.form:
            resultado = traduccion_texto(texto)
        else:
            resultado = 'Acción desconocida'
    return render_template('traductor.html', resultado=resultado)

  
if __name__ == '__main__':
    app.run(debug=True)

