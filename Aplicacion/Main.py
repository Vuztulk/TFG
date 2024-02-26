import tkinter as tk
from tkinter import ttk
from functools import partial
from Clasificacion_Sentimientos import classify_sentiment
from Traduccion import translate_text

def update_translation(input_text, translation_output):
    translated_text = translate_text(input_text.get())
    translation_output.delete(0, tk.END)
    translation_output.insert(0, translated_text)

def update_sentiment(input_text, sentiment_output):
    sentiment_text = classify_sentiment(input_text.get())
    sentiment_output.delete(0, tk.END)
    sentiment_output.insert(0, sentiment_text)

def main():
    root = tk.Tk()
    root.title("Sentiment Classification and Translation")

    # Input Text Entry
    input_label = ttk.Label(root, text="Input Text:")
    input_label.grid(row=0, column=0, padx=5, pady=5)
    input_text = tk.StringVar()
    input_entry = ttk.Entry(root, textvariable=input_text, width=50)
    input_entry.grid(row=0, column=1, padx=5, pady=5)

    # Translation Output Entry
    translation_label = ttk.Label(root, text="Translation:")
    translation_label.grid(row=1, column=0, padx=5, pady=5)
    translation_output = ttk.Entry(root, width=50)
    translation_output.grid(row=1, column=1, padx=5, pady=5)

    # Sentiment Output Entry
    sentiment_label = ttk.Label(root, text="Sentiment:")
    sentiment_label.grid(row=2, column=0, padx=5, pady=5)
    sentiment_output = ttk.Entry(root, width=50)
    sentiment_output.grid(row=2, column=1, padx=5, pady=5)

    # Translate Button
    translate_btn = ttk.Button(root, text="Translate", command=partial(update_translation, input_text, translation_output))
    translate_btn.grid(row=0, column=2, padx=5, pady=5)

    # Classify Sentiment Button
    sentiment_btn = ttk.Button(root, text="Classify Sentiment", command=partial(update_sentiment, input_text, sentiment_output))
    sentiment_btn.grid(row=1, column=2, padx=5, pady=5)

    root.mainloop()

if __name__ == "__main__":
    main()
