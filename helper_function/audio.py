import pyaudio
import wave

# mic idx num
audio = pyaudio.PyAudio()

for idx in range(audio.get_device_count()):
    desc = audio.get_device_info_by_index(idx)
    print(f"DEVICE : {desc['name']}, INDEX : {idx}, RATE : {desc['defaultSampleRate']}")

# audio params
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = r'C:\Project\phishing_bum\phishing-bum\result\audio.wav'

# start recording
stream = audio.open(
    format = FORMAT,
    channels = CHANNELS,
    rate = RATE,
    input = True,
    input_device_index = -1,
    frames_per_buffer = CHUNK
)
print('start recording')

frames = []
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)
print('finished recording')

# stop recording : -> need to be changed : if there is no more input, then stop
stream.stop_stream()
stream.close()
audio.terminate()

# save file
wavefile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wavefile.setnchannels(CHANNELS)
wavefile.setsampwidth(audio.get_sample_size(FORMAT))
wavefile.setframerate(RATE)
wavefile.writeframes(b''.join(frames))
wavefile.close()