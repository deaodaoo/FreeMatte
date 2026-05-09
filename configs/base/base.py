import os
import torch
import torch.nn as nn
import logging
import sys
import time

def setup_devices():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

wait_time = 0
is_shutdown = False
exp_name = "freematte"

def wait_with_countdown(hours_to_wait):
    if hours_to_wait <= 0:
        print("无需等待，直接运行")
        return
    
    total_seconds = int(hours_to_wait * 3600)
    print(f"正在等待程序启动，将在 {hours_to_wait:.2f} 小时后开始...")
    
    try:
        for elapsed in range(total_seconds):
            remaining = total_seconds - elapsed
            hours = remaining // 3600
            minutes = (remaining % 3600) // 60
            seconds = remaining % 60
            
            progress = (elapsed / total_seconds) * 100
            bar_length = 30
            filled_length = int(bar_length * elapsed // total_seconds)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            sys.stdout.write(f'\r[{bar}] {progress:.1f}% | 剩余: {hours:02d}:{minutes:02d}:{seconds:02d}')
            sys.stdout.flush()
            
            time.sleep(1)
    
    except KeyboardInterrupt:
        print(f"\n\n等待被用户中断！总共等待了 {elapsed//3600}:{(elapsed%3600)//60:02d}:{elapsed%60:02d}")
        return
    
    print("\n\n等待结束，开始程序！")



log_dir = "./logs"
log_name = "debug_log.txt"
log_path = os.path.join(log_dir, log_name)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logging.basicConfig(
    level=logging.CRITICAL,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(log_path, mode='w', encoding='utf-8'),
    ]
)

