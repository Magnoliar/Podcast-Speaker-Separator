# 播客语音分离器 (Podcast Speaker Separator)

本项目包含一个 Python 脚本，用于分离双人（或多人）对话播客音频文件中的不同说话人语音。它利用 `pyannote.audio` 库进行说话人日志分析（Speaker Diarization），找出“谁在什么时候说话”，并将每个说话人的语音片段提取到单独的音轨中。

关键特性是：**输出的每个音轨都与原始音频文件具有相同的时间长度，在对应说话人未讲话的时间段填充静音**。这对于需要保持时间轴对齐的应用（如字幕制作、后续分析）非常有用。

## 功能

*   自动检测音频中的说话人及其讲话时间段。
*   为每个检测到的说话人生成一个独立的音频文件。
*   输出的音频文件与原始文件等长，非语音部分填充静音，保持时间轴同步。
*   支持 GPU 加速（如果 CUDA 环境可用）。
*   可以通过命令行参数配置输入/输出路径和 Hugging Face Token。
*   （可选）可以指定预期的说话人数量以辅助模型。

## 环境要求

1.  **Python**: 3.8 或更高版本推荐。
2.  **pip**: Python 包管理器。
3.  **ffmpeg**: 一个用于处理多媒体文件的强大工具。`torchaudio` 可能需要它来加载非 WAV 格式的音频（但强烈建议输入使用 `.wav` 格式）。
    *   **Linux (Ubuntu/Debian):** `sudo apt update && sudo apt install ffmpeg`
    *   **macOS (using Homebrew):** `brew install ffmpeg`
    *   **Windows:** 从 [ffmpeg官网](https://ffmpeg.org/download.html) 下载并将其添加到系统 PATH。
4.  **Hugging Face 账户**: 需要访问 [Hugging Face](https://huggingface.co/) 注册账户。
5.  **Hugging Face 访问令牌 (Token)**:
    *   在 Hugging Face 网站的用户设置 -> Access Tokens 中创建一个具有 'read' 权限的令牌。
    *   你可以通过命令行登录 (`huggingface-cli login`) 或在运行脚本时通过 `--token` 参数提供。
6.  **同意模型使用条款**: `pyannote.audio` 的预训练模型托管在 Hugging Face Hub 上。你需要访问以下模型页面并同意其使用条款：
    *   **主要模型:** [pyannote/speaker-diarization-3.1](https://hf.co/pyannote/speaker-diarization-3.1)
    *   **依赖模型 (例如):** [pyannote/segmentation-3.0](https://hf.co/pyannote/segmentation-3.0) (或模型依赖的其他分割模型)
    *   **依赖模型:** [pyannote/embedding](https://huggingface.co/pyannote/embedding) (或模型依赖的其他嵌入模型)
    *   *注意：具体依赖可能随模型版本变化，如果遇到加载错误，请检查 Hugging Face 上的模型卡片和错误信息。*

## 使用 Google Colab (无需本地安装)

如果你不想在本地设置环境，可以直接在 Google Colab 中运行此项目，利用 Google 提供的免费 GPU 资源。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kY4jXzXvALmsTdYgGaDhzPovulid11Ks?usp=sharing)
*(点击上方按钮直接在 Colab 中打开)*

**在 Colab 中使用的步骤:**

1.  **打开 Notebook:** 点击上面的 "Open In Colab" 按钮。
2.  **设置运行时:**
    *   在 Colab 菜单中，选择 "代码执行程序" (Runtime) -> "更改运行时类型" (Change runtime type)。
    *   在 "硬件加速器" (Hardware accelerator) 下拉菜单中选择 "GPU" (推荐 T4 GPU)。点击 "保存"。
3.  **运行单元格:** 按顺序执行 Notebook 中的代码单元格：
    *   **安装依赖:** 运行第一个单元格来安装所有必需的库。
    *   **Hugging Face 登录:** 运行第二个单元格。它会要求你登录 Hugging Face 账户（需要事先注册并接受模型使用条款，详见"环境要求"部分）。粘贴你的 Hugging Face Access Token。
    *   **上传音频文件:** 运行第三个单元格，点击 "选择文件" 按钮上传你的播客音频文件 (推荐 `.wav` 格式)。
    *   **定义处理函数:** 运行第四个单元格来定义核心的分离逻辑 (只需运行一次)。
    *   **执行分离:** 运行第五个单元格开始处理你的音频文件。你可以在这里设置 `NUM_SPEAKERS` 变量（如果知道说话人数量）。处理过程可能需要一些时间，请耐心等待。
    *   **下载结果:** 处理完成后，运行最后一个单元格。它会将输出目录中的所有分离文件打包成一个 `.zip` 文件，并自动触发浏览器下载。
4.  **检查下载:** 在你的浏览器下载文件夹中查找名为 `separated_audio_colab.zip` (或类似名称) 的文件。解压后即可获得分离后的音频。

**Colab 使用提示:**

*   Colab 会话有时间限制，长时间不活动或总运行时长达到限制后，环境会被重置，上传的文件和安装的库会丢失。
*   确保在运行需要 Token 的单元格之前，已经在 Hugging Face 网站上接受了 `pyannote/speaker-diarization-3.1` 等模型的使用条款。

## 安装与设置

1.  **克隆仓库:**
    ```bash
    git clone https://github.com/Magnoliar/Podcast-Speaker-Separator.git
    cd podcast-speaker-separator
    ```

2.  **创建并激活虚拟环境 (推荐):**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Linux / macOS
    source venv/bin/activate
    ```

3.  **安装 PyTorch:**
    根据你的操作系统和 CUDA 版本（如果使用 GPU），访问 [PyTorch 官网](https://pytorch.org/get-started/locally/) 获取最适合你的安装命令，并执行它。例如（CPU 版本）：
    ```bash
    pip install torch torchvision torchaudio
    ```
    或 CUDA 11.8 版本:
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

4.  **安装其他依赖:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Hugging Face 认证:**
    *   **方法一 (推荐):** 登录 Hugging Face CLI。运行以下命令并粘贴你的 Access Token：
        ```bash
        huggingface-cli login
        ```
    *   **方法二:** 在运行脚本时使用 `--token` 参数传入你的 Access Token。

## 如何使用

通过命令行运行 `separate_speakers.py` 脚本。

**基本用法:**

```bash
python separate_speakers.py -i /path/to/your/podcast.wav -o /path/to/output_directory
```

**参数说明:**

*   `-i, --input`: **必需**. 输入的音频文件路径。强烈推荐使用 `.wav` 格式以获得最佳兼容性。
*   `-o, --output`: **必需**. 用于保存分离后音频文件的目录。如果目录不存在，脚本会尝试创建它。
*   `--token`: **可选**. 你的 Hugging Face Hub 访问令牌。如果已使用 `huggingface-cli login` 登录，则无需提供。
*   `--num-speakers`: **可选**. 整数，指定你期望音频中有多少位说话人。这可以帮助模型在某些情况下（例如语音重叠较多或声音相似时）提高准确性，但不是必需的。

**示例:**

```bash
# 处理名为 "episode1.wav" 的文件，结果保存到 "separated_audio" 目录
python separate_speakers.py -i episode1.wav -o separated_audio

# 处理文件，并明确告知模型有 2 位说话人
python separate_speakers.py -i my_interview.wav -o output_files --num-speakers 2

# 处理文件，并使用 --token 参数提供 HF Token
python separate_speakers.py -i meeting.mp3 -o separated_meeting --token hf_YOUR_TOKEN_HERE
```

**输出:**

脚本执行成功后，会在指定的输出目录 (`-o` 参数指定的目录) 中生成多个 `.wav` 文件。每个文件对应一个检测到的说话人，文件名格式通常为：

`<原始文件名>_speaker_SPEAKER_XX.wav`

例如，对于输入 `episode1.wav`，可能会生成：

*   `episode1_speaker_SPEAKER_00.wav`
*   `episode1_speaker_SPEAKER_01.wav`

这些文件的长度与原始音频相同，包含了对应说话人的语音片段，其他时间段则为静音。

## 注意事项

*   **处理时间**: 音频文件的时长和你的硬件（CPU/GPU）会显著影响处理时间。长音频可能需要较长时间。
*   **准确性**: `pyannote.audio` 是一个强大的库，但在非常嘈杂的环境、说话人声音非常相似或语音重叠严重的情况下，分离结果可能不完美。
*   **内存消耗**: 处理非常长的音频文件可能会消耗大量内存（尤其是 RAM 和 GPU 显存）。
*   **音频格式**: 虽然脚本可能能处理 `ffmpeg` 支持的其他格式（如 MP3），但强烈建议将输入音频预先转换为 `.wav` 格式（例如，16kHz 单声道 PCM）以获得最佳效果和兼容性。

## 致谢

本项目基于强大的 [pyannote.audio](https://github.com/pyannote/pyannote-audio) 库实现。感谢其开发者和社区。
