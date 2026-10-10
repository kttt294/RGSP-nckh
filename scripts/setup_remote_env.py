import paramiko
import time
import sys

def run_ssh_command(ssh, cmd, title=""):
    print(f"\n==================================================")
    print(f"[*] {title or cmd}")
    print(f"==================================================")
    stdin, stdout, stderr = ssh.exec_command(cmd, get_pty=True)
    
    # Stream output in real time
    while True:
        line = stdout.readline()
        if not line:
            break
        print(line, end="")
        sys.stdout.flush()

    status = stdout.channel.recv_exit_status()
    print(f"[+] Exit code: {status}")
    if status != 0:
        err = stderr.read().decode().strip()
        if err:
            print(f"[-] Stderr: {err}")
    return status

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print("[*] Connecting to aiotser (100.91.149.28)...")
    ssh.connect('100.91.149.28', port=22, username='hoangnv', password='Hoang123@', timeout=20)
    print("[+] Connected successfully!")

    # Step 1: Create project directory and cache directory on /storage (374GB free)
    run_ssh_command(ssh, "mkdir -p /home/hoangnv/rgsp_nckh /storage/hoangnv/.cache/huggingface", "Tạo thư mục dự án và cache Hugging Face")

    # Step 2: Check if rs_solve conda env already exists
    stdin, stdout, stderr = ssh.exec_command("/home/hoangnv/miniconda3/bin/conda env list")
    envs = stdout.read().decode()
    if "rs_solve " in envs or "/envs/rs_solve" in envs:
        print("[+] Môi trường 'rs_solve' đã tồn tại!")
    else:
        # Create conda env with python 3.10
        status = run_ssh_command(ssh, "/home/hoangnv/miniconda3/bin/conda create -n rs_solve python=3.10 -y", "Tạo môi trường ảo Conda 'rs_solve' (Python 3.10)")
        if status != 0:
            print("[-] Tạo môi trường conda thất bại!")
            ssh.close()
            sys.exit(1)

    # Step 3: Install torch + torchvision with CUDA 12.4 inside rs_solve
    install_torch_cmd = """
/home/hoangnv/miniconda3/envs/rs_solve/bin/pip install --upgrade pip
/home/hoangnv/miniconda3/envs/rs_solve/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
"""
    run_ssh_command(ssh, install_torch_cmd, "Cài đặt PyTorch với CUDA 12.4 vào môi trường rs_solve")

    # Step 4: Install required libraries: transformers, accelerate, bitsandbytes, qwen-vl-utils, etc.
    install_pkgs_cmd = """
/home/hoangnv/miniconda3/envs/rs_solve/bin/pip install transformers accelerate bitsandbytes pillow tqdm qwen-vl-utils torchvision
"""
    run_ssh_command(ssh, install_pkgs_cmd, "Cài đặt Transformers, BitsAndBytes (8-bit), Accelerate")

    # Step 5: Verify installation and GPU access from rs_solve
    verify_cmd = """
/home/hoangnv/miniconda3/envs/rs_solve/bin/python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device:', torch.cuda.get_device_name(0))
    print('VRAM Total (GB):', torch.cuda.get_device_properties(0).total_memory / 1e9)
"
"""
    run_ssh_command(ssh, verify_cmd, "Kiểm tra xác thực PyTorch và GPU CUDA trong môi trường rs_solve")

    ssh.close()
    print("\n[+] HOÀN TẤT THIẾT LẬP MÔI TRƯỜNG rs_solve TRÊN SERVER!")

if __name__ == "__main__":
    main()
