from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/enviar', methods=['POST'])
def recibir_texto():
    if request.method == 'POST':
        texto = request.form['texto']
        # Aquí puedes hacer lo que quieras con el texto, por ejemplo, imprimirlo
        print("Texto recibido:", texto)
        return "Texto recibido: " + texto

if __name__ == '__main__':
    app.run(debug=True)
