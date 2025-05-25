# import os
# import cv2
# import numpy as np
# import librosa
# import tensorflow as tf
# from flask import Flask, request, render_template
# from pydub import AudioSegment
# from werkzeug.utils import secure_filename
# # from flask import Flask, request, render_template

# # ✅ Force TensorFlow to use CPU (to avoid CUDA errors)
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# # ✅ Define custom layers BEFORE loading models
# class Conv2Plus1D(tf.keras.layers.Layer):
#     def __init__(self, filters, kernel_size, padding="same", **kwargs):
#         super().__init__(**kwargs)
#         self.seq = tf.keras.Sequential([
#             tf.keras.layers.Conv3D(filters=filters, kernel_size=(1, kernel_size[1], kernel_size[2]), padding=padding),
#             tf.keras.layers.Conv3D(filters=filters, kernel_size=(kernel_size[0], 1, 1), padding=padding)
#         ])

#     def call(self, x):
#         return self.seq(x)

# class ResizeVideo(tf.keras.layers.Layer):
#     def __init__(self, height, width, **kwargs):
#         super().__init__(**kwargs)
#         self.height = height
#         self.width = width
#         self.resizing_layer = tf.keras.layers.Resizing(self.height, self.width)

#     def call(self, video):
#         shape = tf.shape(video)
#         batch_size, num_frames = shape[0], shape[1]
#         video_reshaped = tf.reshape(video, (-1, shape[2], shape[3], shape[4]))
#         resized_frames = self.resizing_layer(video_reshaped)
#         output = tf.reshape(resized_frames, (batch_size, num_frames, self.height, self.width, shape[4]))
#         return output

# class ResidualMain(tf.keras.layers.Layer):
#     def __init__(self, filters, kernel_size, **kwargs):
#         super().__init__(**kwargs)
#         self.seq = tf.keras.Sequential([
#             Conv2Plus1D(filters=filters, kernel_size=kernel_size, padding='same'),
#             tf.keras.layers.LayerNormalization(),
#             tf.keras.layers.ReLU(),
#             Conv2Plus1D(filters=filters, kernel_size=kernel_size, padding='same'),
#             tf.keras.layers.LayerNormalization()
#         ])

#     def call(self, x):
#         return self.seq(x)

# class Project(tf.keras.layers.Layer):
#     def __init__(self, units, **kwargs):
#         super().__init__(**kwargs)
#         self.seq = tf.keras.Sequential([
#             tf.keras.layers.Dense(units),
#             tf.keras.layers.LayerNormalization()
#         ])

#     def call(self, x):
#         return self.seq(x)

# # ✅ Load pre-trained models with custom layers
# video_model = tf.keras.models.load_model(
#     r"C:\Users\yogit\Downloads\lstm\r_2_1_d_3d_cnn_model_final_combined.h5",
#     custom_objects={'Conv2Plus1D': Conv2Plus1D, 'ResizeVideo': ResizeVideo, 'ResidualMain': ResidualMain, 'Project': Project},
#     compile=False
# )

# audio_model = tf.keras.models.load_model(
#     r"C:\Users\yogit\Downloads\lstm\updated_model.h5",
#     compile=False
# )

# app = Flask(__name__)
# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # ✅ Function to extract frames from video
# def extract_frames(video_path, n_frames=10):
#     cap = cv2.VideoCapture(video_path)
#     frames = []
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     frame_interval = max(frame_count // n_frames, 1)

#     for i in range(n_frames):
#         cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_interval)
#         ret, frame = cap.read()
#         if ret:
#             frame = cv2.resize(frame, (224, 224))
#             frame = frame.astype('float32') / 255.0
#             frames.append(frame)

#     cap.release()
#     return np.array(frames).reshape(1, n_frames, 224, 224, 3)

# # ✅ Function to extract audio from video
# def extract_audio(video_path, output_audio_path):
#     try:
#         audio = AudioSegment.from_file(video_path)
#         audio.export(output_audio_path, format="wav")
#         return output_audio_path
#     except Exception as e:
#         print(f"Audio extraction failed: {e}")
#         return None

# # ✅ Function to process audio for deepfake detection
# # def process_audio(audio_path):
# #     try:
# #         y, sr = librosa.load(audio_path, sr=16000)
# #         mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
# #         return np.expand_dims(mfcc, axis=0)
# #     except Exception as e:
# #         print(f"Audio processing error: {e}")
# #         return None

# # Function to process audio for deepfake detection
# def process_audio(audio_path, max_length=500):
#     y, sr = librosa.load(audio_path, sr=16000)
#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

#     # 🔹 Ensure fixed feature size (40, 500)
#     if mfcc.shape[1] < max_length:
#         # Pad if too short
#         mfcc = np.pad(mfcc, ((0, 0), (0, max_length - mfcc.shape[1])), mode='constant')
#     else:
#         # Truncate if too long
#         mfcc = mfcc[:, :max_length]

#     # Reshape to match the model's input format (batch_size, 40, 500, 1)
#     mfcc = np.expand_dims(mfcc, axis=-1)  # Add channel dimension
#     mfcc = np.expand_dims(mfcc, axis=0)   # Add batch dimension

#     return mfcc

# from flask import Flask, request, render_template, send_from_directory
# import os

# app = Flask(__name__)
# UPLOAD_FOLDER = "uploads"
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# @app.route('/', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         file = request.files['file']
#         if file:
#             filename = file.filename
#             file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(file_path)

#             violence_result = "No Violence"
#             audio_result = "Real Audio"
#             file_type = None

#             if filename.endswith(('.mp4', '.avi', '.mov')):
#                 file_type = "video"
#                 frames = extract_frames(file_path)
#                 violence_prediction = video_model.predict(frames)
#                 violence_result = "Violence Detected" if violence_prediction[0][0] > 0.5 else "No Violence"

#                 audio_path = os.path.join(app.config['UPLOAD_FOLDER'], "extracted_audio.wav")
#                 extract_audio(file_path, audio_path)
#                 audio_features = process_audio(audio_path)
#                 audio_prediction = audio_model.predict(audio_features)
#                 audio_result = "Fake Audio" if audio_prediction[0][0] > 0.5 else "Real Audio"

#             elif filename.endswith(('.wav', '.mp3')):
#                 file_type = "audio"
#                 audio_features = process_audio(file_path)
#                 audio_prediction = audio_model.predict(audio_features)
#                 audio_result = "Fake Audio" if audio_prediction[0][0] > 0.5 else "Real Audio"

#             file_url = f"/uploads/{filename}"  # URL for file visualization

#             return render_template('result.html', violence_result=violence_result, audio_result=audio_result, file_url=file_url, file_type=file_type)

#     return render_template('upload.html')

# if __name__ == '__main__':
#     app.run(debug=True)



import os
import cv2
import numpy as np
import librosa
import tensorflow as tf
from flask import Flask, request, render_template, send_from_directory
from pydub import AudioSegment
from werkzeug.utils import secure_filename

# ✅ Force TensorFlow to use CPU (to avoid CUDA errors)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ✅ Define Custom Layers (Must Be Defined Before Loading Model)
class Conv2Plus1D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding="same", **kwargs):
        super().__init__(**kwargs)
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Conv3D(filters=filters, kernel_size=(1, kernel_size[1], kernel_size[2]), padding=padding),
            tf.keras.layers.Conv3D(filters=filters, kernel_size=(kernel_size[0], 1, 1), padding=padding)
        ])

    def call(self, x):
        return self.seq(x)

class ResizeVideo(tf.keras.layers.Layer):
    def __init__(self, height, width, **kwargs):
        super().__init__(**kwargs)
        self.height = height
        self.width = width
        self.resizing_layer = tf.keras.layers.Resizing(self.height, self.width)

    def call(self, video):
        shape = tf.shape(video)
        batch_size, num_frames = shape[0], shape[1]
        video_reshaped = tf.reshape(video, (-1, shape[2], shape[3], shape[4]))
        resized_frames = self.resizing_layer(video_reshaped)
        output = tf.reshape(resized_frames, (batch_size, num_frames, self.height, self.width, shape[4]))
        return output

class ResidualMain(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.seq = tf.keras.Sequential([
            Conv2Plus1D(filters=filters, kernel_size=kernel_size, padding='same'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.ReLU(),
            Conv2Plus1D(filters=filters, kernel_size=kernel_size, padding='same'),
            tf.keras.layers.LayerNormalization()
        ])

    def call(self, x):
        return self.seq(x)

class Project(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(units),
            tf.keras.layers.LayerNormalization()
        ])

    def call(self, x):
        return self.seq(x)

# ✅ Load Pre-trained Models
video_model = tf.keras.models.load_model(
    r"C:\Users\yogit\Downloads\lstm\r_2_1_d_3d_cnn_model_final_combined.h5",
    custom_objects={'Conv2Plus1D': Conv2Plus1D, 'ResizeVideo': ResizeVideo, 'ResidualMain': ResidualMain, 'Project': Project},
    compile=False
)

audio_model = tf.keras.models.load_model(
    r"C:\Users\yogit\Downloads\lstm\updated_model.h5",
    compile=False
)

# ✅ Flask App Setup
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ✅ Extract Frames from Video for Violence Detection
def extract_frames(video_path, n_frames=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(frame_count // n_frames, 1)

    for i in range(n_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_interval)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (224, 224))
            frame = frame.astype('float32') / 255.0
            frames.append(frame)

    cap.release()
    return np.array(frames).reshape(1, n_frames, 224, 224, 3)

# ✅ Extract Audio from Video
def extract_audio(video_path, output_audio_path):
    try:
        audio = AudioSegment.from_file(video_path)
        audio.export(output_audio_path, format="wav")
        return output_audio_path
    except Exception as e:
        print(f"Audio extraction failed: {e}")
        return None

# ✅ Process Audio for Deepfake Detection
def process_audio(audio_path, max_length=500):
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

        # 🔹 Ensure Fixed Feature Size (40, 500)
        if mfcc.shape[1] < max_length:
            mfcc = np.pad(mfcc, ((0, 0), (0, max_length - mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :max_length]

        # Reshape to match the model's input format (batch_size, 40, 500, 1)
        mfcc = np.expand_dims(mfcc, axis=-1)  # Add channel dimension
        mfcc = np.expand_dims(mfcc, axis=0)   # Add batch dimension

        return mfcc
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

# ✅ Flask Route: Serve Uploaded Files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ✅ Flask Route: Upload Page
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            violence_result = "No Violence"
            audio_result = "No Audio Detected"
            file_type = None

            if filename.endswith(('.mp4', '.avi', '.mov')):
                file_type = "video"

                # 🔹 Step 1: Detect Violence in Video
                frames = extract_frames(file_path)
                violence_prediction = video_model.predict(frames)
                violence_result = "Violence Detected" if violence_prediction[0][0] > 0.5 else "No Violence"

                # 🔹 Step 2: Extract & Check Audio (Only If Present)
                audio_path = os.path.join(app.config['UPLOAD_FOLDER'], "extracted_audio.wav")
                audio_file = extract_audio(file_path, audio_path)

                if audio_file and os.path.exists(audio_file):
                    audio_features = process_audio(audio_file)
                    if audio_features is not None:
                        audio_prediction = audio_model.predict(audio_features)
                        audio_result = "Fake Audio" if audio_prediction[0][0] > 0.5 else "Real Audio"

            elif filename.endswith(('.wav', '.mp3')):
                file_type = "audio"
                audio_features = process_audio(file_path)

                if audio_features is not None:
                    audio_prediction = audio_model.predict(audio_features)
                    audio_result = "Fake Audio" if audio_prediction[0][0] > 0.5 else "Real Audio"

            file_url = f"/uploads/{filename}"  # URL for file visualization

            return render_template('result.html', violence_result=violence_result, audio_result=audio_result, file_url=file_url, file_type=file_type)

    return render_template('upload.html')

# ✅ Run Flask App
if __name__ == '__main__':
    app.run(debug=True, port=5001)
