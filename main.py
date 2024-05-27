from flask import Flask, request, render_template, send_from_directory
import os
import torch
import torchaudio
from denoiser import pretrained
from denoiser.dsp import convert_audio

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load the pretrained model
model = pretrained.dns64().cuda()

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Load the input WAV file
        wav, sr = torchaudio.load(filepath)

        # Convert the audio if necessary
        wav = convert_audio(wav.cuda(), sr, model.sample_rate, model.chin)

        # Denoise the audio
        with torch.no_grad():
            denoised = model(wav[None])[0]

        # Save the denoised audio to a new file
        output_filepath = os.path.join(OUTPUT_FOLDER, 'denoised_' + file.filename)
        torchaudio.save(output_filepath, denoised.data.cpu(), model.sample_rate)

        return render_template('result.html', filename='denoised_' + file.filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=False)
