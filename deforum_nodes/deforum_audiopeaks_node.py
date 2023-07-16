
def get_peak_values(mp3_path, threshold, number_of_samples, distribution=None):
    import librosa

    # Load the audio file
    audio, sr = librosa.load(mp3_path)

    # Calculate the magnitude of the audio signal
    magnitude = np.abs(librosa.stft(audio))

    # Find the peak values
    peak_values = np.max(magnitude, axis=0)

    # Find the low peaks by subtracting a threshold from the peak values
    low_peaks = peak_values - threshold
    low_peaks[low_peaks < 0] = 0

    # Normalize the peak values between 0.00 and 1.00
    normalized_peaks = peak_values / np.max(peak_values)
    normalized_lows = low_peaks / np.max(low_peaks)

    mix = []

    # Select the desired number of samples for peaks
    if len(normalized_peaks) > number_of_samples:
        indices = np.argsort(normalized_peaks)[::-1][:number_of_samples]
        normalized_peaks = normalized_peaks[indices]

    # Select the desired number of samples for lows
    if len(normalized_lows) > number_of_samples:
        indices = np.argsort(normalized_lows)[::-1][:number_of_samples]
        normalized_lows = normalized_lows[indices]

    # Compute the frame numbers based on the number of samples
    if distribution == "linear":
        frame_numbers = np.linspace(0, number_of_samples - 1, len(normalized_peaks), dtype=int)
    elif distribution == "logarithmic":
        frame_numbers = np.round(np.geomspace(1, number_of_samples, len(normalized_peaks))).astype(int) - 1
    else:
        frame_numbers = np.arange(len(normalized_peaks))

    for i, peak in enumerate(normalized_peaks.tolist()):
        frame_number = frame_numbers[i]
        mix.append(f"Peak {frame_number}:({peak:.2f})")

    for i, low in enumerate(normalized_lows.tolist()):
        frame_number = frame_numbers[i]
        mix.append(f"Low {frame_number}:({low:.2f})")

    return ", ".join(mix)


import numpy as np

def generate_keyframes(mp3_path, fps):
    import librosa
    # Load the audio file
    audio, sr = librosa.load(mp3_path)

    # Calculate the duration of the audio in seconds
    audio_duration = len(audio) / sr

    # Calculate the number of frames based on the FPS and audio duration
    total_frames = int(audio_duration * fps)

    # Calculate the hop length for the STFT (Short-Time Fourier Transform)
    hop_length = len(audio) // total_frames

    # Calculate the STFT of the audio signal
    stft = librosa.stft(audio, hop_length=hop_length)

    # Calculate the spectral energy for each frame
    spectral_energy = librosa.feature.rms(S=stft)

    # Calculate the peak energy value for each frame
    peak_energy = np.max(spectral_energy, axis=0)

    # Normalize the peak energy values between 0.00 and 10.00
    normalized_peak_energy = librosa.util.normalize(peak_energy) * 10.0

    # Generate the keyframes as a string
    keyframes = ""
    for frame_num, peak_value in enumerate(normalized_peak_energy):
        time = frame_num / fps
        keyframes += f"{frame_num}:({peak_value:.2f}),"

    return keyframes.rstrip("\n")  # Remove the trailing newline character



import numpy as np
from qtpy import QtWidgets, QtCore

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget

OP_NODE_DEFORUM_PEAKS = get_next_opcode()
class DeforumAudioPeaksWidget(QDMNodeContentWidget):
    set_peaks_signal = QtCore.Signal(str)
    def initUI(self):
        # Create a label to display the image
        self.createUI()
        self.create_main_layout(grid=1)
    def createUI(self):
        self.path_edit = self.create_line_edit("Path")
        self.peaks = self.create_line_edit("Peaks")



@register_node(OP_NODE_DEFORUM_PEAKS)
class DeforumAudioNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/bg.png"
    op_code = OP_NODE_DEFORUM_PEAKS
    op_title = "Deforum Peaks"
    content_label_objname = "deforum_audio_peaks_node"
    category = "aiNodes Deforum/DeForum"


    def __init__(self, scene):
        super().__init__(scene, inputs=[1], outputs=[1])

    def initInnerClasses(self):
        self.content = DeforumAudioPeaksWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.height = 200
        self.grNode.width = 280
        self.content.eval_signal.connect(self.evalImplementation)
        self.content.set_peaks_signal.connect(self.set_peak)

    #@QtCore.Slot(str)
    def set_peak(self, peak):
        self.content.peaks.setText(peak)



    #@QtCore.Slot()
    def evalImplementation_thread(self, index=0):

        path = self.content.path_edit.text()
        peaks = generate_keyframes(path, 25)

        print(peaks)

        self.content.set_peaks_signal.emit(peaks)
        return None


    #@QtCore.Slot(object)
    def onWorkerFinished(self, result):
        self.busy = False
        #super().onWorkerFinished(None)
        if len(self.getOutputs(0)) > 0:
            self.executeChild(output_index=0)

