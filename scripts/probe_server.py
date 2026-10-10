import paramiko

def probe():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('100.91.149.28', port=22, username='hoangnv', password='Hoang123@', timeout=15)

    commands = [
        "df -h /home/hoangnv",
        "ls -la /home/hoangnv",
        "find /home/hoangnv -maxdepth 3 -type d 2>/dev/null | grep -iE 'data|dataset|dior|dota|xview|visdrone' || true"
    ]

    for c in commands:
        print(f"=== Command: {c} ===")
        stdin, stdout, stderr = ssh.exec_command(c)
        print(stdout.read().decode().strip())

    ssh.close()

if __name__ == "__main__":
    probe()
