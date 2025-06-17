import io
import os
import time
import requests
import shutil
import tempfile
import zipfile
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
# CORS(app)
CORS(app, origins=["http://localhost:8000"])

@app.route('/generate_nifti', methods=['POST'])
def generate_nifti():
    # output_path = "output/output.nii.gz"
    output_path = "output/3d.obj"
    return jsonify({'nifti_url': output_path})

@app.route('/nifti/<path:filename>')
def serve_nifti(filename):
    return send_from_directory('.', filename)

if __name__ == '__main__':
    app.run(debug=True)


