import tkinter as tk
from tkinter import messagebox
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Ensure stopwords are available
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

# Load the saved model and vectorizer
try:
    model = joblib.load('logistic_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
except FileNotFoundError:
    print("Error: Model files not found. Please run train_and_save.py first.")
    exit()

def process_text():
    input_text = text_input.get("1.0", tk.END).strip()
    if not input_text:
        messagebox.showwarning("Warning", "Please enter some text.")
        return
        
    # Preprocess and predict
    cleaned = preprocess_text(input_text)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    
    # Update the result text box
    result = "Fake Review (CG)" if prediction == "CG" else "Real Review (OR)"
    
    # We populate the first text box with our algorithm's result. 
    result_box_1.config(state=tk.NORMAL)
    result_box_1.delete(0, tk.END)
    result_box_1.insert(0, result)
    result_box_1.config(state=tk.DISABLED)

# Create the main window
root = tk.Tk()
root.title("Fake Review Detection System")
root.geometry("600x500")

# Title Label
tk.Label(root, text="Fake Product Review Identification", font=("Helvetica", 16, "bold")).pack(pady=10)

# Input Text Box
tk.Label(root, text="Enter Product Review:", font=("Helvetica", 12)).pack(anchor="w", padx=20)
text_input = tk.Text(root, height=5, width=65)
text_input.pack(padx=20, pady=5)

# Process Button
process_btn = tk.Button(root, text="Process", font=("Helvetica", 12, "bold"), bg="#4CAF50", fg="white", command=process_text)
process_btn.pack(pady=10)

# 5 Output Text Boxes (as specified in PDF)
tk.Label(root, text="Algorithm 1 Result (Logistic Regression):", font=("Helvetica", 10)).pack(anchor="w", padx=20)
result_box_1 = tk.Entry(root, width=70, state=tk.DISABLED, disabledbackground="white", disabledforeground="black")
result_box_1.pack(padx=20, pady=2)

tk.Label(root, text="Algorithm 2 Result:", font=("Helvetica", 10)).pack(anchor="w", padx=20)
result_box_2 = tk.Entry(root, width=70, state=tk.DISABLED)
result_box_2.pack(padx=20, pady=2)

tk.Label(root, text="Algorithm 3 Result:", font=("Helvetica", 10)).pack(anchor="w", padx=20)
result_box_3 = tk.Entry(root, width=70, state=tk.DISABLED)
result_box_3.pack(padx=20, pady=2)

tk.Label(root, text="Algorithm 4 Result:", font=("Helvetica", 10)).pack(anchor="w", padx=20)
result_box_4 = tk.Entry(root, width=70, state=tk.DISABLED)
result_box_4.pack(padx=20, pady=2)

tk.Label(root, text="Algorithm 5 Result:", font=("Helvetica", 10)).pack(anchor="w", padx=20)
result_box_5 = tk.Entry(root, width=70, state=tk.DISABLED)
result_box_5.pack(padx=20, pady=2)

# Run the Tkinter event loop
root.mainloop()
