"""Week 5: Model serving utilities — LoRA merge, GGUF conversion, Ollama integration."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _detect_adapter_format(adapter_path: str) -> str:
    """Return 'hf', 'mlx', or 'unknown' based on adapter_config.json shape."""
    cfg_file = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.isfile(cfg_file):
        return "unknown"
    with open(cfg_file) as f:
        cfg = json.load(f)
    if "peft_type" in cfg or "base_model_name_or_path" in cfg:
        return "hf"
    if "fine_tune_type" in cfg or "lora_parameters" in cfg or "model" in cfg:
        return "mlx"
    return "unknown"


def _merge_lora_mlx(base_model_path: str, adapter_path: str, output_path: str) -> str:
    """Merge an MLX-format LoRA adapter via mlx_lm.fuse. Produces an HF-format dir."""
    print(f"[model_serve] Detected MLX-format adapter; using mlx_lm.fuse to merge.")
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Try Python API first (mlx_lm >= 0.20)
    try:
        from mlx_lm import fuse  # type: ignore
        if hasattr(fuse, "main"):
            print(f"[model_serve] Calling mlx_lm.fuse.main(...)")
            argv = [
                "--model", base_model_path,
                "--adapter-path", adapter_path,
                "--save-path", output_path,
                "--dequantize",  # produce fp16/bf16 weights, not 4-bit, for downstream HF/GGUF use
            ]
            old_argv = sys.argv
            try:
                sys.argv = ["mlx_lm.fuse"] + argv
                fuse.main()
            finally:
                sys.argv = old_argv
            print(f"[model_serve] mlx_lm.fuse complete -> {output_path}")
            return output_path
    except ImportError:
        pass
    except Exception as e:
        print(f"[model_serve] mlx_lm.fuse Python API failed: {e}; falling back to subprocess.")

    # Subprocess fallback
    cmd = [
        sys.executable, "-m", "mlx_lm.fuse",
        "--model", base_model_path,
        "--adapter-path", adapter_path,
        "--save-path", output_path,
        "--dequantize",
    ]
    print(f"[model_serve] Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(
                f"mlx_lm.fuse failed (exit {result.returncode}):\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
        print(result.stdout[-500:] if result.stdout else "")
        print(f"[model_serve] mlx_lm.fuse complete -> {output_path}")
        return output_path
    except FileNotFoundError:
        raise RuntimeError(
            "mlx_lm not installed. Install: pip install mlx-lm  (Apple Silicon only)"
        )


def _merge_lora_hf(base_model_path: str, adapter_path: str, output_path: str, torch_dtype: str) -> str:
    """Merge an HF PEFT-format LoRA adapter via PeftModel.merge_and_unload."""
    try:
        import torch
    except ImportError:
        raise ImportError("Install pytorch: pip install torch")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError("Install transformers: pip install transformers")

    try:
        from peft import PeftModel
    except ImportError:
        raise ImportError("Install peft: pip install peft")

    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    dtype = dtype_map.get(torch_dtype, torch.float16)

    print(f"[model_serve] Loading base model with dtype={torch_dtype}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=dtype, device_map="auto", trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    print("[model_serve] Loading LoRA adapter (HF PEFT format)...")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("[model_serve] Merging and unloading LoRA weights...")
    merged_model = model.merge_and_unload()

    Path(output_path).mkdir(parents=True, exist_ok=True)
    print(f"[model_serve] Saving merged model to {output_path}...")
    merged_model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    print(f"[model_serve] Merge complete -> {output_path}")
    return output_path


def merge_lora(
    base_model_path: str,
    adapter_path: str,
    output_path: str,
    torch_dtype: str = "float16",
) -> str:
    """
    Merge a LoRA adapter into the base model and save as a HF-format dir.

    Auto-detects adapter format:
    - HF PEFT adapter (peft_type / base_model_name_or_path) -> PeftModel.merge_and_unload()
    - MLX-LoRA adapter (fine_tune_type / lora_parameters)   -> mlx_lm.fuse (Apple Silicon only)
    """
    print(f"[model_serve] Merging LoRA adapter into base model...")
    print(f"[model_serve] Base: {base_model_path}")
    print(f"[model_serve] Adapter: {adapter_path}")
    print(f"[model_serve] Output: {output_path}")

    fmt = _detect_adapter_format(adapter_path)
    if fmt == "mlx":
        return _merge_lora_mlx(base_model_path, adapter_path, output_path)
    if fmt == "hf":
        return _merge_lora_hf(base_model_path, adapter_path, output_path, torch_dtype)
    raise ValueError(
        f"[model_serve] Cannot determine adapter format at '{adapter_path}'. "
        f"Expected adapter_config.json with either 'peft_type' (HF PEFT) or "
        f"'fine_tune_type'/'lora_parameters' (MLX-LoRA)."
    )


def convert_to_gguf(
    hf_model_path: str,
    output_path: str,
    quant: str = "Q4_K_M",
) -> str | None:
    """Convert HF model to GGUF using llama.cpp's convert_hf_to_gguf.py. Returns output_path or None."""
    print(f"[model_serve] Converting {hf_model_path} to GGUF (quant={quant})...")

    # Search common llama.cpp locations
    llama_cpp_candidates = [
        os.path.expanduser("~/llama.cpp"),
        os.path.expanduser("~/repos/llama.cpp"),
        "/opt/llama.cpp",
        "/usr/local/llama.cpp",
        os.path.join(os.getcwd(), "llama.cpp"),
    ]

    convert_script: str | None = None
    quantize_bin: str | None = None

    for candidate in llama_cpp_candidates:
        script = os.path.join(candidate, "convert_hf_to_gguf.py")
        if os.path.exists(script):
            convert_script = script
            quantize_bin = os.path.join(candidate, "build", "bin", "llama-quantize")
            if not os.path.exists(quantize_bin):
                quantize_bin = os.path.join(candidate, "quantize")
            break

    if convert_script is None:
        print(
            "[model_serve] llama.cpp not found. To install:\n"
            "  git clone https://github.com/ggerganov/llama.cpp\n"
            "  cd llama.cpp && cmake -B build && cmake --build build --config Release\n"
            "  pip install -r requirements.txt\n"
            "Searched locations:\n" + "\n".join(f"  {c}" for c in llama_cpp_candidates)
        )
        return None

    print(f"[model_serve] Found llama.cpp convert script: {convert_script}")
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Generate GGUF file name
    model_name = Path(hf_model_path).name or "model"
    gguf_base_path = os.path.join(output_path, f"{model_name}-f16.gguf")
    gguf_final_path = os.path.join(output_path, f"{model_name}-{quant}.gguf")

    # Step 1: Convert to F16 GGUF
    print("[model_serve] Step 1: Converting to F16 GGUF...")
    convert_cmd = [
        sys.executable,
        convert_script,
        hf_model_path,
        "--outfile", gguf_base_path,
        "--outtype", "f16",
    ]
    try:
        result = subprocess.run(
            convert_cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"[model_serve] Conversion output:\n{result.stdout[-500:]}")
    except subprocess.CalledProcessError as e:
        print(f"[model_serve] GGUF conversion failed (exit code {e.returncode}):\n{e.stderr[-500:]}")
        return None
    except Exception as e:
        print(f"[model_serve] Unexpected error during conversion: {e}")
        return None

    # Step 2: Quantize
    if quant != "f16" and quant != "F16" and quantize_bin and os.path.exists(quantize_bin):
        print(f"[model_serve] Step 2: Quantizing to {quant}...")
        quant_cmd = [quantize_bin, gguf_base_path, gguf_final_path, quant]
        try:
            result = subprocess.run(quant_cmd, capture_output=True, text=True, check=True)
            print(f"[model_serve] Quantization output:\n{result.stdout[-500:]}")
            # Clean up F16 intermediate
            if os.path.exists(gguf_base_path) and gguf_base_path != gguf_final_path:
                os.remove(gguf_base_path)
                print(f"[model_serve] Removed intermediate F16 file")
        except subprocess.CalledProcessError as e:
            print(f"[model_serve] Quantization failed (exit code {e.returncode}): {e.stderr[-300:]}")
            print(f"[model_serve] Using F16 GGUF instead: {gguf_base_path}")
            gguf_final_path = gguf_base_path
        except Exception as e:
            print(f"[model_serve] Quantize error: {e}. Using F16 fallback.")
            gguf_final_path = gguf_base_path
    else:
        if quant not in ("f16", "F16"):
            print(f"[model_serve] llama-quantize binary not found. Skipping quantization. Using F16 GGUF.")
        gguf_final_path = gguf_base_path

    print(f"[model_serve] GGUF ready at: {gguf_final_path}")
    return gguf_final_path


def make_ollama_modelfile(
    gguf_path: str,
    output_path: str = "outputs/ollama_modelfile.txt",
    system_prompt: str = "You are a helpful assistant trained on resume data.",
    model_name: str = "hw5-finetuned",
) -> str:
    """
    Generate an Ollama Modelfile with Qwen2.5 ChatML template + sane runtime params.

    Without the TEMPLATE directive, Ollama feeds prompts raw to the model — Qwen
    never sees the <|im_start|>...<|im_end|> structure it was trained on, so stop
    tokens never trigger and inference appears to hang forever.

    Sets num_ctx=2048 (default 32768 wastes massive KV-cache RAM) and num_predict=300
    so generation is bounded.
    """
    print(f"[model_serve] Generating Ollama Modelfile for {model_name}...")

    abs_path = os.path.abspath(gguf_path)

    # Qwen2.5 ChatML template. Ollama's Go-template syntax: {{ .System }} {{ .Prompt }} {{ .Response }}
    template = (
        '"""'
        '{{ if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}'
        '{{ if .Prompt }}<|im_start|>user\n{{ .Prompt }}<|im_end|>\n{{ end }}'
        '<|im_start|>assistant\n{{ .Response }}<|im_end|>\n'
        '"""'
    )

    modelfile_content = (
        f"FROM {abs_path}\n\n"
        f"TEMPLATE {template}\n\n"
        f"SYSTEM \"\"\"\n{system_prompt}\n\"\"\"\n\n"
        # Sampling
        f"PARAMETER temperature 0.7\n"
        f"PARAMETER top_p 0.9\n"
        f"PARAMETER repeat_penalty 1.1\n"
        # Context + output bounds — without these Ollama allocates 32k-token KV-cache by default
        f"PARAMETER num_ctx 2048\n"
        f"PARAMETER num_predict 300\n"
        # Stop tokens — Qwen2.5's ChatML end markers
        f"PARAMETER stop \"<|im_end|>\"\n"
        f"PARAMETER stop \"<|im_start|>\"\n"
        f"PARAMETER stop \"<|endoftext|>\"\n"
    )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(modelfile_content)
    print(f"[model_serve] Modelfile saved to {output_path}")
    print(f"[model_serve] To register with Ollama: ollama create {model_name} -f {output_path}")

    return modelfile_content


def ollama_create_and_test(
    model_name: str,
    modelfile_path: str,
    test_prompt: str = "Who are you and what can you help me with?",
) -> str | None:
    """Run 'ollama create' then send one test prompt via Ollama API. Returns response or None."""
    print(f"[model_serve] Creating Ollama model '{model_name}' from {modelfile_path}...")

    # Check ollama CLI availability
    try:
        check = subprocess.run(["ollama", "--version"], capture_output=True, text=True, timeout=10)
        if check.returncode != 0:
            raise FileNotFoundError()
        print(f"[model_serve] Ollama version: {check.stdout.strip()}")
    except FileNotFoundError:
        print(
            "[model_serve] Ollama not found. To install:\n"
            "  macOS/Linux: curl -fsSL https://ollama.ai/install.sh | sh\n"
            "  Windows: https://ollama.ai/download\n"
            "  Then run: ollama serve"
        )
        return None

    # Create the model
    print(f"[model_serve] Running: ollama create {model_name} -f {modelfile_path}")
    try:
        create_result = subprocess.run(
            ["ollama", "create", model_name, "-f", modelfile_path],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if create_result.returncode != 0:
            print(f"[model_serve] ollama create failed:\n{create_result.stderr[-500:]}")
            return None
        print(f"[model_serve] ollama create output:\n{create_result.stdout[-300:]}")
    except subprocess.TimeoutExpired:
        print("[model_serve] ollama create timed out after 5 minutes")
        return None
    except Exception as e:
        print(f"[model_serve] ollama create error: {e}")
        return None

    # Test via Ollama REST API. First call after `ollama create` triggers model load
    # from disk into memory which can take 30-120s on Mac for a 0.5B-1B model.
    print(f"[model_serve] Testing model with prompt: {test_prompt[:60]}...")
    print(f"[model_serve] (first inference loads the model — may take up to 5 min on Mac)")
    try:
        import urllib.request
        import urllib.error

        payload = json.dumps({
            "model": model_name,
            "prompt": test_prompt,
            "stream": False,
        }).encode()

        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=300) as resp:
            response_data = json.loads(resp.read().decode())
            response_text = response_data.get("response", "")

        print(f"[model_serve] Ollama response: {response_text[:200]}")
        return response_text

    except urllib.error.URLError as e:
        print(f"[model_serve] Could not reach Ollama server at localhost:11434: {e}")
        print("[model_serve] Make sure Ollama is running: ollama serve")
        return None
    except (TimeoutError, OSError) as e:
        # socket.timeout is a TimeoutError on Python 3.10+ and OSError on older
        msg = str(e) or type(e).__name__
        print(f"[model_serve] Ollama inference timed out after 5 min: {msg}")
        print(f"[model_serve] Model '{model_name}' was CREATED successfully — only the test prompt timed out.")
        print(f"[model_serve] Try manually: ollama run {model_name} \"{test_prompt}\"")
        print(f"[model_serve] If first run is slow, the model is loading from disk (one-time cost).")
        return f"[create_succeeded_but_test_timeout] Run: ollama run {model_name}"
    except Exception as e:
        print(f"[model_serve] Ollama API call failed: {e}")
        return None
