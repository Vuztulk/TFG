from flask import Flask, render_template, request
import requests

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        texto = request.form.get('texto')
        response = requests.post('http://other-ip/process', data={'text': texto})
        processed_text = response.text
        return render_template('index.html', processed_text=processed_text)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
