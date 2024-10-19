




# import numpy as np
# import cv2
# import os
# from flask import Flask, render_template, request, redirect, url_for, flash
# from PIL import Image
# import pytesseract
# import pdfplumber 
# from spellchecker import SpellChecker

# spell = SpellChecker() # For handling PDFs

# app = Flask(__name__)
# app.secret_key = '88026a7e14a5615e2bc6539ca5273d11'  # Use your secret key

# # Path to the Tesseract executable
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust to your path

# # Route for the homepage
# @app.route('/')
# def index():
#     return render_template('index.html')

# # Route to handle file upload
# @app.route('/upload', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         # Handle file upload (same as before)
#         if 'file' not in request.files:
#             flash("No file part in the request.")
#             return redirect(url_for('upload_file'))

#         file = request.files['file']

#         if file.filename == '':
#             flash("No file selected.")
#             return redirect(url_for('upload_file'))

#         if file and allowed_file(file.filename):
#             file_path = os.path.join('uploads', file.filename)
#             file.save(file_path)

#             # Determine file type and process accordingly
#             extracted_text = process_file(file_path)

#             # Save the result to a text file
#             output_txt_path = os.path.join('uploads', 'output_text.txt')  # Save in the uploads directory
#             with open(output_txt_path, 'w', encoding='utf-8') as f:
#                 f.write(extracted_text)

#             # Send the text back as a downloadable file or display
#             return render_template('result.html', extracted_text=extracted_text)
#         else:
#             flash("Invalid file type. Only PDF and images are supported.")
#             return redirect(url_for('upload_file'))

#     # For GET request, show the upload form
#     return render_template('upload.html')  # Make sure this template exists
#  # Assuming you have a template for the upload form

# def clear_image(img):
#     brightness = 10 
# # Adjusts the contrast by scaling the pixel values by 2.3 
#     contrast = 2.3  
#     image2 = cv2.addWeighted(img, contrast, np.zeros(img.shape, img.dtype), 0, brightness) 
#     return image2

# def allowed_file(filename):
#     ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def process_file(file_path):
#     """Process the uploaded file and extract text."""
#     if file_path.endswith('.pdf'):
#         return extract_text_from_pdf(file_path)
#     elif file_path.endswith(('.png', '.jpg', '.jpeg')):
#         return extract_text_from_image(file_path)
#     else:
#         return "Unsupported file format."

# # def preprocess_image(image):
# #     # Resize or normalize the image based on your model input
# #     image = image.resize((128, 32))  # Example size; adjust as necessary
# #     image = np.array(image) / 255.0  # Normalize to [0, 1]
# #     return image


# def remove_noise(image_path):
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     denoised = cv2.medianBlur(image, 3)  # Apply median blur
#     return Image.fromarray(denoised)

# def dilate_image(image_path):
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     kernel = np.ones((1, 1), np.uint8)
#     dilated = cv2.dilate(image, kernel, iterations=1)
#     return Image.fromarray(dilated)

# def preprocess_image(image_path):
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
#     _, thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)  # Binarize
#     return Image.fromarray(thresh)

# # def image_to_text(image_path):
# #     preprocessed_image = preprocess_image(image_path)
# #     text = pytesseract.image_to_string(preprocessed_image)
# #     return text

# # def decode_predictions(predictions):
# #     # Convert the model output to text
# #     # This depends on how your model outputs predictions (e.g., using argmax)
# #     decoded_text = ''  # Implement decoding logic
# #     return decoded_text


# def extract_text_from_image(image_path):
#     """Extract text from an image using OCR."""
#     image = Image.open(image_path)
#     if image.mode in ['P', 'RGBA']:  # Handle transparency or palette images
#         image = image.convert('RGB')
#         image = clear_image(image)
#         image = preprocess_image(image)
#         image= dilate_image(image)
#         image=remove_noise(image)
#          # Implement this function based on your model's requirements
#         # input_data = np.expand_dims(image, axis=0)
#     ocr_result = pytesseract.image_to_string(image)
#     # predictions = model.predict(input_data)
#     # extracted_text = decode_predictions(predictions)
#     return ocr_result

# # def autocorrect_text(text):
# #     """Correct the text using a spell checker."""
# #     words = text.split()
# #     corrected_words = [spell.candidates(word) for word in words]
# #     # Replace words with the most common correct spelling
# #     corrected_text = ' '.join([next(iter(candidates), word) for word, candidates in zip(words, corrected_words)])
# #     return corrected_text

# # def autocorrect_text(text):
# #     """Correct the text using a spell checker."""
# #     words = text.split()
# #     corrected_words = []

# #     for word in words:
# #         # Get the candidates for the word
# #         candidates = spell.candidates(word)
# #         if candidates:
# #             # Take the most common candidate
# #             corrected_word = next(iter(candidates))
# #         else:
# #             # If no candidates, keep the original word
# #             corrected_word = word
# #         corrected_words.append(corrected_word)

# #     # Join corrected words into a single string
# #     corrected_text = ' '.join(corrected_words)
# #     return corrected_text


# def extract_text_from_pdf(pdf_path):
#     """Extract text from a PDF using pdfplumber."""
#     text = ''
#     with pdfplumber.open(pdf_path) as pdf:
#         for page in pdf.pages:
#             text += page.extract_text() or ''  # Handle None values
#     return text

# if __name__ == '__main__':
#     if not os.path.exists('uploads'):
#         os.makedirs('uploads')
#     app.run(debug=True)



























# import numpy as np
# import os
# from flask import Flask, render_template, request, redirect, url_for, flash
# from PIL import Image
# import pytesseract
# import pdfplumber
# from spellchecker import SpellChecker
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize, sent_tokenize
# import nltk

# # Ensure NLTK resources are downloaded
# nltk.download('stopwords')
# nltk.download('punkt')

# spell = SpellChecker()  # For handling spell-checking

# app = Flask(__name__)
# app.secret_key = '88026a7e14a5615e2bc6539ca5273d11'  # Use your secret key

# # Path to the Tesseract executable
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust to your path

# # Route for the homepage
# @app.route('/')
# def index():
#     return render_template('index.html')

# # Route to handle file upload
# @app.route('/upload', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             flash("No file part in the request.")
#             return redirect(url_for('upload_file'))

#         file = request.files['file']

#         if file.filename == '':
#             flash("No file selected.")
#             return redirect(url_for('upload_file'))

#         if file and allowed_file(file.filename):
#             file_path = os.path.join('uploads', file.filename)
#             file.save(file_path)

#             # Determine file type and process accordingly
#             extracted_text = process_file(file_path)

#             # Summarize the extracted text
#             summarized_text = summarize_text(extracted_text)

#             # Save the result to a text file
#             output_txt_path = os.path.join('uploads', 'output_text.txt')  # Save in the uploads directory
#             with open(output_txt_path, 'w', encoding='utf-8') as f:
#                 f.write(summarized_text)

#             # Send the summarized text back as a downloadable file or display
#             return render_template('result.html', extracted_text=summarized_text)
#         else:
#             flash("Invalid file type. Only PDF and images are supported.")
#             return redirect(url_for('upload_file'))

#     return render_template('upload.html')  # Make sure this template exists

# def extract_text_from_image(image_path):
#     """Extract text from an image using OCR."""
#     image = Image.open(image_path)

#     # Convert image to RGB if it has an alpha channel or palette
#     if image.mode in ['P', 'RGBA']:
#         image = image.convert('RGB')
        
#     # Perform OCR
#     ocr_result = pytesseract.image_to_string(image)
#     return ocr_result

# def allowed_file(filename):
#     """Check if the uploaded file is allowed."""
#     ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def process_file(file_path):
#     """Process the uploaded file and extract text."""
#     if file_path.endswith('.pdf'):
#         return extract_text_from_pdf(file_path)
#     elif file_path.endswith(('.png', '.jpg', '.jpeg')):
#         return extract_text_from_image(file_path)
#     else:
#         return "Unsupported file format."

# def extract_text_from_pdf(pdf_path):
#     """Extract text from a PDF using pdfplumber."""
#     text = ''
#     with pdfplumber.open(pdf_path) as pdf:
#         for page in pdf.pages:
#             text += page.extract_text() or ''  # Handle None values
#     return text

# def summarize_text(text):
#     """Summarize the provided text."""
#     stopWords = set(stopwords.words("english"))
#     words = word_tokenize(text)

#     # Create a frequency table to keep the score of each word
#     freqTable = {}
#     for word in words:
#         word = word.lower()
#         if word in stopWords:
#             continue
#         if word in freqTable:
#             freqTable[word] += 1
#         else:
#             freqTable[word] = 1

#     # Tokenize the text into sentences
#     sentences = sent_tokenize(text)

#     # Create a dictionary to keep the score of each sentence
#     sentenceValue = {}
#     for sentence in sentences:
#         for word, freq in freqTable.items():
#             if word in sentence.lower():
#                 if sentence in sentenceValue:
#                     sentenceValue[sentence] += freq
#                 else:
#                     sentenceValue[sentence] = freq

#     # Calculate the average sentence value
#     sumValues = sum(sentenceValue.values())
#     average = int(sumValues / len(sentenceValue)) if sentenceValue else 0

#     # Store sentences into our summary based on their score
#     summary = ''
#     for sentence in sentences:
#         if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.2 * average)):
#             summary += " " + sentence

#     return summary.strip()

# if __name__ == '__main__':
#     if not os.path.exists('uploads'):
#         os.makedirs('uploads')
#     app.run(debug=True)




























# import numpy as np
# import os
# from flask import Flask, render_template, request, redirect, url_for, flash, session
# from PIL import Image
# import pytesseract
# import pdfplumber
# from spellchecker import SpellChecker
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize, sent_tokenize
# import nltk

# # Ensure NLTK resources are downloaded
# nltk.download('stopwords')
# nltk.download('punkt')

# spell = SpellChecker()  # For handling spell-checking

# app = Flask(__name__)
# app.secret_key = '88026a7e14a5615e2bc6539ca5273d11'  # Use your secret key

# # Path to the Tesseract executable
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust to your path

# # Route for the homepage
# @app.route('/')
# def index():
#     return render_template('index.html')

# # Route to handle file upload
# @app.route('/upload', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             flash("No file part in the request.")
#             return redirect(url_for('upload_file'))

#         file = request.files['file']

#         if file.filename == '':
#             flash("No file selected.")
#             return redirect(url_for('upload_file'))

#         if file and allowed_file(file.filename):
#             file_path = os.path.join('uploads', file.filename)
#             file.save(file_path)

#             # Determine file type and process accordingly
#             extracted_text = process_file(file_path)

#             # Store extracted text in session
#             session['extracted_text'] = extracted_text  # Use session directly

#             return render_template('upload.html', extracted_text=extracted_text)  # Pass extracted text to the template
#         else:
#             flash("Invalid file type. Only PDF and images are supported.")
#             return redirect(url_for('upload_file'))

#     return render_template('upload.html')  # Make sure this template exists

# def extract_text_from_image(image_path):
#     """Extract text from an image using OCR."""
#     image = Image.open(image_path)

#     # Convert image to RGB if it has an alpha channel or palette
#     if image.mode in ['P', 'RGBA']:
#         image = image.convert('RGB')
        
#     # Perform OCR
#     ocr_result = pytesseract.image_to_string(image)
#     return ocr_result

# def allowed_file(filename):
#     """Check if the uploaded file is allowed."""
#     ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def process_file(file_path):
#     """Process the uploaded file and extract text."""
#     if file_path.endswith('.pdf'):
#         return extract_text_from_pdf(file_path)
#     elif file_path.endswith(('.png', '.jpg', '.jpeg')):
#         return extract_text_from_image(file_path)
#     else:
#         return "Unsupported file format."

# def extract_text_from_pdf(pdf_path):
#     """Extract text from a PDF using pdfplumber."""
#     text = ''
#     with pdfplumber.open(pdf_path) as pdf:
#         for page in pdf.pages:
#             text += page.extract_text() or ''  # Handle None values
#     return text

# def summarize_text(text):
#     """Summarize the provided text."""
#     stopWords = set(stopwords.words("english"))
#     words = word_tokenize(text)

#     # Create a frequency table to keep the score of each word
#     freqTable = {}
#     for word in words:
#         word = word.lower()
#         if word in stopWords:
#             continue
#         if word in freqTable:
#             freqTable[word] += 1
#         else:
#             freqTable[word] = 1

#     # Tokenize the text into sentences
#     sentences = sent_tokenize(text)

#     # Create a dictionary to keep the score of each sentence
#     sentenceValue = {}
#     for sentence in sentences:
#         for word, freq in freqTable.items():
#             if word in sentence.lower():
#                 if sentence in sentenceValue:
#                     sentenceValue[sentence] += freq
#                 else:
#                     sentenceValue[sentence] = freq

#     # Calculate the average sentence value
#     sumValues = sum(sentenceValue.values())
#     average = int(sumValues / len(sentenceValue)) if sentenceValue else 0

#     # Store sentences into our summary based on their score
#     summary = ''
#     for sentence in sentences:
#         if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.2 * average)):
#             summary += " " + sentence

#     return summary.strip()

# @app.route('/summarize', methods=['POST'])
# def summarize():
#     """Summarize the extracted text."""
#     extracted_text = session.get('extracted_text', '')  # Get the extracted text from the session
#     summarized_text = summarize_text(extracted_text)  # Summarize the text

#     return render_template('result.html', summarized_text=summarized_text)  # Return the result page

# if __name__ == '__main__':
#     if not os.path.exists('uploads'):
#         os.makedirs('uploads')
#     app.run(debug=True)





import numpy as np
import os
from flask import Flask, render_template, request, redirect, url_for, flash, session
from PIL import Image
import pytesseract
import pdfplumber
from spellchecker import SpellChecker
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk

# Ensure NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('punkt')

spell = SpellChecker()  # For handling spell-checking

app = Flask(__name__)
app.secret_key = '88026a7e14a5615e2bc6539ca5273d11'  # Use your secret key

# Path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust to your path

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle file upload
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash("No file part in the request.")
            return redirect(url_for('upload_file'))

        file = request.files['file']

        if file.filename == '':
            flash("No file selected.")
            return redirect(url_for('upload_file'))

        if file and allowed_file(file.filename):
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)

            # Determine file type and process accordingly
            extracted_text = process_file(file_path)

            # Store extracted text in session
            session['extracted_text'] = extracted_text  # Use session directly

            return render_template('upload.html', extracted_text=extracted_text)  # Pass extracted text to the template
        else:
            flash("Invalid file type. Only PDF and images are supported.")
            return redirect(url_for('upload_file'))

    return render_template('upload.html')  # Make sure this template exists

def extract_text_from_image(image_path):
    """Extract text from an image using OCR."""
    image = Image.open(image_path)

    # Convert image to RGB if it has an alpha channel or palette
    if image.mode in ['P', 'RGBA']:
        image = image.convert('RGB')
        
    # Perform OCR
    ocr_result = pytesseract.image_to_string(image)
    return ocr_result

def allowed_file(filename):
    """Check if the uploaded file is allowed."""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_file(file_path):
    """Process the uploaded file and extract text."""
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith(('.png', '.jpg', '.jpeg')):
        return extract_text_from_image(file_path)
    else:
        return "Unsupported file format."

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF using pdfplumber."""
    text = ''
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ''  # Handle None values
    return text

def summarize_text(text):
    """Summarize the provided text."""
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text)

    # Create a frequency table to keep the score of each word
    freqTable = {}
    for word in words:
        word = word.lower()
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Create a dictionary to keep the score of each sentence
    sentenceValue = {}
    for sentence in sentences:
        for word, freq in freqTable.items():
            if word in sentence.lower():
                if sentence in sentenceValue:
                    sentenceValue[sentence] += freq
                else:
                    sentenceValue[sentence] = freq

    # Calculate the average sentence value
    sumValues = sum(sentenceValue.values())
    average = int(sumValues / len(sentenceValue)) if sentenceValue else 0

    # Store sentences into our summary based on their score
    summary = ''
    for sentence in sentences:
        if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.2 * average)):
            summary += " " + sentence

    return summary.strip()

@app.route('/summarize', methods=['POST'])
def summarize():
    """Summarize the extracted text."""
    extracted_text = session.get('extracted_text', '')  # Get the extracted text from the session
    summarized_text = summarize_text(extracted_text)  # Summarize the text

    return render_template('result.html', extracted_text=extracted_text, summarized_text=summarized_text)  # Return the result page

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
