#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torchaudio
from pyannote.audio import Pipeline
from pyannote.core import Segment
import os
import numpy as np
import argparse
import logging
import sys

# --- 配置日志记录 ---
# 配置日志记录器，使其同时输出到控制台和文件
log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
root_logger = logging.getLogger()

# 设置日志级别 (例如 INFO, DEBUG)
root_logger.setLevel(logging.INFO)

# 控制台处理器
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
root_logger.addHandler(console_handler)

# (可选) 文件处理器，取消注释以启用日志文件记录
# log_filename = "speaker_separation.log"
# file_handler = logging.FileHandler(log_filename)
# file_handler.setFormatter(log_formatter)
# root_logger.addHandler(file_handler)


# --- 主要处理函数 ---
def separate_speakers(audio_path, output_dir, hf_token=None, num_speakers=None):
    """
    使用 pyannote.audio 分离音频文件中的说话人语音，并保持原始时间轴。

    Args:
        audio_path (str): 输入音频文件的路径。
        output_dir (str): 保存分离后音频文件的目录。
        hf_token (str, optional): Hugging Face Hub 访问令牌。默认为 None (依赖 huggingface-cli login)。
        num_speakers (int, optional): 预期的说话人数量。如果提供，传递给 diarization pipeline。默认为 None。
    """
    logging.info("===== 开始处理 =====")
    logging.info(f"输入文件: {audio_path}")
    logging.info(f"输出目录: {output_dir}")
    if num_speakers:
        logging.info(f"指定说话人数量: {num_speakers}")
    if hf_token:
        logging.info("使用提供的 Hugging Face Token")
    else:
        logging.info("未提供 HF Token，将依赖 'huggingface-cli login' 的缓存认证")


    # 0. 检查 GPU 是否可用
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"检测到 CUDA 设备: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logging.info("未检测到 CUDA 设备，将使用 CPU (处理速度较慢)")

    # 1. 加载 pyannote.audio 的说话人日志流水线
    #    使用最新的稳定模型 pyannote/speaker-diarization-3.1
    #    用户需要事先在 Hugging Face Hub 同意此模型的使用条款
    #    https://hf.co/pyannote/speaker-diarization-3.1
    #    以及依赖的 segmentation 模型条款
    #    https://hf.co/pyannote/segmentation-3.0 (或类似依赖)
    pipeline_name = "pyannote/speaker-diarization-3.1"
    try:
        logging.info(f"加载说话人日志模型: {pipeline_name}...")
        pipeline = Pipeline.from_pretrained(pipeline_name, use_auth_token=hf_token)
        pipeline.to(device)
        logging.info("模型加载完成。")
    except Exception as e:
        logging.error(f"加载 pyannote 模型失败: {e}", exc_info=True)
        logging.error("请确保：")
        logging.error("  1. 已安装 'pyannote.audio' 库 (请参阅 requirements.txt)。")
        logging.error("  2. 已通过 'huggingface-cli login' 登录，或通过 --token 参数提供了有效的 Hugging Face Token。")
        logging.error(f"  3. 已在 Hugging Face 网站上同意了 '{pipeline_name}' 及相关依赖模型的使用条款。")
        return False # 指示处理失败

    # 2. 执行说话人日志 (Diarization)
    logging.info(f"正在分析音频文件: {audio_path}...")
    try:
        # 准备 pipeline 的参数
        pipeline_kwargs = {}
        if num_speakers is not None:
            pipeline_kwargs['num_speakers'] = num_speakers

        # 执行流水线
        diarization = pipeline(audio_path, **pipeline_kwargs)
        logging.info("说话人日志分析完成。")
    except Exception as e:
        logging.error(f"执行说话人日志时出错: {e}", exc_info=True)
        if "CUDA out of memory" in str(e):
             logging.error("CUDA 显存不足！请尝试处理较短的音频，或使用 CPU (如果当前未使用)。")
        elif "torchaudio" in str(e).lower():
             logging.error("可能是音频文件格式或加载问题。推荐使用 WAV 格式。")
        return False # 指示处理失败

    # 3. 加载原始音频文件
    logging.info("加载原始音频数据...")
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        # 确保音频是单声道 (如果原始音频是立体声，只取第一个声道)
        if waveform.shape[0] > 1:
            logging.warning(f"检测到 {waveform.shape[0]} 个声道，将使用第一个声道进行处理。")
            waveform = waveform[0:1, :] # 取第一个声道，并保持二维形状 [1, N]

        # 移动到正确的设备 (GPU 或 CPU)
        waveform = waveform.to(device)

        logging.info(f"音频加载完成。采样率: {sample_rate} Hz, 时长: {waveform.shape[1] / sample_rate:.2f} 秒, 设备: {waveform.device}")
    except Exception as e:
        logging.error(f"加载音频文件 {audio_path} 失败: {e}", exc_info=True)
        return False # 指示处理失败

    # 4. 创建输出目录 (如果不存在)
    try:
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"确保输出目录 '{output_dir}' 已存在。")
    except OSError as e:
        logging.error(f"创建输出目录 '{output_dir}' 失败: {e}", exc_info=True)
        return False

    # 5. 根据日志结果分离音频
    speakers = diarization.labels()
    num_detected_speakers = len(speakers)
    logging.info(f"检测到 {num_detected_speakers} 位说话人: {', '.join(speakers)}")
    if num_speakers is not None and num_detected_speakers != num_speakers:
        logging.warning(f"警告：检测到的说话人数量 ({num_detected_speakers}) 与指定的数量 ({num_speakers}) 不符。将按检测结果继续。")
    elif num_detected_speakers == 0:
        logging.warning("警告：未检测到任何说话人活动。请检查输入音频或模型。")
        return True # 技术上处理完成，但无有效输出

    # 为每个说话人创建一个与原始音频等长的空白 (静音) 音轨 (在同一设备上)
    separated_waveforms = {speaker: torch.zeros_like(waveform) for speaker in speakers}

    logging.info("开始根据时间戳分离音轨...")
    total_segments = sum(1 for _ in diarization.itertracks(yield_label=True))
    processed_segments = 0

    # 遍历 diarization 结果中的每个语音片段
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        # 将时间 (秒) 转换为采样点索引
        start_sample = int(turn.start * sample_rate)
        end_sample = int(turn.end * sample_rate)

        # 安全检查，防止计算出的索引超出音频实际长度
        end_sample = min(end_sample, waveform.shape[1])
        start_sample = min(start_sample, end_sample) # 确保 start 不会大于 end

        # 从原始音频中提取当前说话人的语音片段 (确保在同一设备操作)
        # 注意：直接在 GPU 上进行切片和赋值，避免不必要的 CPU<->GPU 数据传输
        try:
            if speaker in separated_waveforms:
                separated_waveforms[speaker][:, start_sample:end_sample] = waveform[:, start_sample:end_sample]
            else:
                logging.warning(f"在日志中找到说话人 {speaker}，但该说话人不在初始列表中。跳过此片段。") # 理论上不应发生

            processed_segments += 1
            # 打印进度，避免长时间运行无反馈 (使用 logging.debug 或调整打印频率)
            if processed_segments % 100 == 0 or processed_segments == total_segments:
                 logging.info(f"  处理进度: {processed_segments}/{total_segments} 个语音片段")

        except Exception as e:
            logging.error(f"处理片段 Speaker {speaker} [{turn.start:.2f}s - {turn.end:.2f}s] 时出错: {e}", exc_info=True)
            # 可以选择跳过此片段或中止处理
            continue # 跳过这个片段

    logging.info("音轨分离完成。")

    # 6. 保存分离后的音频文件
    logging.info("正在保存分离后的文件...")
    output_paths = {} # 用于存储每个说话人对应的输出文件路径
    output_base_name = os.path.splitext(os.path.basename(audio_path))[0] # 获取不带扩展名的原始文件名

    save_success = True
    for speaker, wave in separated_waveforms.items():
        # 构建输出文件名
        output_filename = os.path.join(output_dir, f"{output_base_name}_speaker_{speaker}.wav")
        try:
            # 保存前将 Tensor 移动到 CPU
            wave_cpu = wave.cpu()
            # 保存处理后的波形数据为 WAV 文件
            torchaudio.save(output_filename, wave_cpu, sample_rate)
            output_paths[speaker] = output_filename # 记录保存的路径
            logging.info(f"  已保存: {output_filename}")
        except Exception as e:
            logging.error(f"保存文件 {output_filename} 时出错: {e}", exc_info=True)
            save_success = False # 标记至少有一个文件保存失败

    if save_success:
        logging.info("所有分离文件已成功保存。")
        logging.info("===== 处理成功完成！ =====")
        return True
    else:
        logging.error("部分或全部分离文件保存失败。")
        logging.info("===== 处理完成，但有保存错误 =====")
        return False


# --- 主程序入口 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 pyannote.audio 分离音频文件中的说话人语音，并保持原始时间轴。")
    parser.add_argument("-i", "--input", required=True, help="输入音频文件的路径 (推荐使用 WAV 格式)。")
    parser.add_argument("-o", "--output", required=True, help="保存分离后音频文件的输出目录。")
    parser.add_argument("--token", help="Hugging Face Hub 访问令牌。如果未提供，则依赖 'huggingface-cli login' 的认证。")
    parser.add_argument("--num-speakers", type=int, help="（可选）预期的说话人数量。有助于提高某些情况下的准确性。")
    # 添加一个静默模式选项 (可选)
    # parser.add_argument("-q", "--quiet", action="store_true", help="只输出错误信息。")

    args = parser.parse_args()

    # (可选) 根据 quiet 参数调整日志级别
    # if args.quiet:
    #     root_logger.setLevel(logging.ERROR)
    #     # 可能需要移除 console_handler 或设置其级别为 ERROR

    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        logging.error(f"错误：输入文件 '{args.input}' 不存在。")
        sys.exit(1) # 退出脚本，返回错误码

    # 运行主处理函数
    success = separate_speakers(
        audio_path=args.input,
        output_dir=args.output,
        hf_token=args.token,
        num_speakers=args.num_speakers
    )

    # 根据处理结果返回退出码
    if success:
        sys.exit(0) # 成功
    else:
        sys.exit(1) # 失败
