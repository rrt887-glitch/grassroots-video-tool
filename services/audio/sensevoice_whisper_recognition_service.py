import subprocess
import numpy as np
from typing import List
from tools.utils import must_have_value

# 条件导入sherpa_onnx
try:
    import sherpa_onnx
    SHERPA_ONNX_AVAILABLE = True
except ImportError:
    SHERPA_ONNX_AVAILABLE = False
    sherpa_onnx = None

class SenseVoiceRecognitionResult:
    def __init__(self, text, begin_time, end_time):
        self.text = text
        self.begin_time = begin_time
        self.end_time = end_time

    def __str__(self):
        return f"{self.text} {self.begin_time} {self.end_time}"


class SenseVoiceRecognitionService:
    def __init__(self):
        super().__init__()
        if not SHERPA_ONNX_AVAILABLE:
            raise ImportError("sherpa-onnx依赖不可用，SenseVoice语音识别功能无法使用")
        
        self.model_path = "sensevoice/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.onnx"
        must_have_value(self.model_path, "请设置 SenseVoice 模型路径")
        self.tokens_path = "sensevoice/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt"
        must_have_value(self.tokens_path, "请设置 SenseVoice tokens 路径")

    def process(self, audioFile, language) -> List[SenseVoiceRecognitionResult]:
        result_list = []

        # 创建 SenseVoice 识别器
        recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=self.model_path,
            tokens=self.tokens_path,
            num_threads=2,  # 可以根据需要调整线程数
            use_itn=True,
            debug=False,
        )

        # 使用 ffmpeg 将音频文件转换为 16kHz 16bit 单声道 PCM 格式
        ffmpeg_cmd = [
            "ffmpeg",
            "-i", audioFile,
            "-f", "s16le",
            "-acodec", "pcm_s16le",
            "-ac", "1",
            "-ar", "16000",
            "-",
        ]

        process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        data = process.stdout.read()

        # 将音频数据转换为 float32 格式
        samples = np.frombuffer(data, dtype=np.int16)
        samples = samples.astype(np.float32) / 32768

        # 创建识别流并处理音频数据
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, samples)
        recognizer.decode_stream(stream)

        # 获取识别结果
        result = stream.result.text

        # 假设整个音频的起止时间为 0 到音频长度
        result_list.append(SenseVoiceRecognitionResult(result, 0, len(samples) / 16000))

        return result_list