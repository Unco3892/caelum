"""
LLM inference and clarification / answer generation for CAELUM.

Supports local HuggingFace models (e.g. Qwen2.5-7B-Instruct) and remote
API providers (Gemini, Groq, OpenAI) via OpenAI-compatible interface.
"""

import gc
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates (inline defaults — can be overridden via file loading)
# ---------------------------------------------------------------------------

TRIVIAQA_CLARIFICATION_PROMPT = """\
**Objective**
In this task, you will receive a question. Your goal is to generate multiple versions of the question that convey the same meaning as the original one.

**Important Rules**
1. Ensure that each rephrasing of the question is distinct from the others.
2. Ensure that all rephrasings of the question are semantically equivalent to the original question.
3. Provide {n_clarifications} different rephrasings of the question.

**Output Format**
Your output should follow this format:
### Rephrasings:
#1 [Your rephrased question]
#2 [Another rephrased question]
#3 [Yet another rephrased question]
#4 [A fourth rephrasing of the question]
#5 [A fifth rephrasing of the question]

**Question**: {question}
"""

AMBIGQA_CLARIFICATION_PROMPT = """\
**Objective**
Analyze the given question for ambiguities. If the question is ambiguous, your task is to clarify it by interpreting the ambiguous concepts, specifying necessary conditions, or using other methods. Provide as much different clarifications as possible.
An ambiguous question is a question that has different correct answers, depending on individual interpretations. Your clarifications are supposed to remove any ambiguity in the question so every clarified question will have a single possible correct answer. These ambiguities can arise from various factors, including but not limited to:
1. Ambiguous references to entities in the question.
2. Multiple properties of objects/entities in the question leading to different interpretations.
3. Ambiguities due to unclear timestamps.
4. Ambiguities stemming from unclear locations.
5. Multiple valid answer types based on the question.
6. References to undefined or underspecified entities in the question.

**Important Rules**
1. Perform detailed analyses before concluding whether the question is clear or ambiguous. In the analyses, you can rely on your general knowledge to anticipate possible correct answers and interpretations of the question.
2. Output clarifications in the specified format. Do not include possible answers in the clarifications. The clarifications should be only more precise rephrasings of the same question.
3. For each ambiguous question, you are to provide at least two distinct rephrasings that resolve these ambiguities. By "rephrasing," we mean you should reformulate the question to be clear and direct, eliminating any possible ambiguity without altering the original intent of the question. You should not seek further information or produce a binary (yes-no) question as a result of the clarification. Instead, you must create a direct question (wh-question) that aims to obtain a specific answer.
4. Do not provide more than {n_clarifications} clarifications for an ambiguous question.
5. Do not provide placeholders in your clarifications. They must be fully contained explicit questions.
6. Do not add explanations within the clarifications of the questions. All your reasoning, analyses and explanation should be contained in the Analyses section only.

**Output Format**
Your output should follow this format:
### Analyses:
[Think step-by-step to reason on the clarity of the question, possible answers and interpretations. After that, output your judgement on whether the question is ambiguous or not]

### Clarifications:
#1 [First rephrased question]
#2 [Second rephrased question]
#3 [Third rephrased question]
...

If the question is clear and unambiguous, simply output:
### Clarifications:
#1 No clarification needed.

**Question**: {question}
"""

ANSWER_PROMPT = """\
Answer the following question concisely in one or two sentences.

Question: {question}
Answer:"""


# ---------------------------------------------------------------------------
# HuggingFace local model wrapper
# ---------------------------------------------------------------------------

class HFModel:
    """Local HuggingFace model wrapper for generation.

    Supports both pure CausalLM models and VLM/conditional generation models
    (e.g., Mistral 3 with vision encoder) — VLMs are used text-only.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        max_new_tokens: int = 256,
        use_4bit: Optional[bool] = None,
    ):
        logger.info("Loading model %s …", model_name)

        # Fine-grained FP8 checkpoints route matmuls through DeepGEMM, whose
        # JIT needs a full CUDA toolkit on the host. On toolkit-less nodes,
        # CAELUM_FP8_FORCE_TRITON=1 makes the DeepGEMM loader raise
        # ImportError so transformers falls back to its universal Triton
        # finegrained-fp8 kernel (slower, but JIT-free).
        if os.environ.get("CAELUM_FP8_FORCE_TRITON", "").strip().lower() in {
            "1", "true", "yes", "on"
        }:
            try:
                from transformers.integrations import finegrained_fp8 as _fp8

                def _deepgemm_disabled() -> None:
                    raise ImportError(
                        "DeepGEMM disabled via CAELUM_FP8_FORCE_TRITON"
                    )

                _fp8._load_deepgemm_kernel = _deepgemm_disabled
                logger.info("FP8: DeepGEMM disabled, using Triton fallback")
            except Exception:  # pragma: no cover - transformers layout drift
                logger.warning("Could not disable DeepGEMM; continuing")

        # Optional dtype override: DeepGEMM/Triton fp8 kernels only emit
        # bf16/fp32 outputs, so fp16-loaded activations trip a dtype
        # assertion at the first forward.
        dtype_env = os.environ.get("CAELUM_HF_DTYPE", "").strip().lower()
        if dtype_env in {"bfloat16", "bf16"}:
            dtype = torch.bfloat16
            logger.info("dtype override via CAELUM_HF_DTYPE: bfloat16")
        elif dtype_env in {"float16", "fp16"}:
            dtype = torch.float16

        # Load tokenizer — try AutoTokenizer first, fall back to AutoProcessor for VLMs
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self._is_processor = False
        except Exception:
            from transformers import AutoProcessor
            self.tokenizer = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self._is_processor = True
            logger.info("Using AutoProcessor (VLM tokenizer) for %s", model_name)

        # Auto-detect 4-bit quantization for large models. Pre-quantized repos
        # such as GPTQ/AWQ/FP8 should use their own loading config unless the
        # caller explicitly forces bitsandbytes with CAELUM_HF_USE_4BIT=1.
        if use_4bit is None:
            forced_4bit = os.environ.get("CAELUM_HF_USE_4BIT", "").strip().lower()
            if forced_4bit in {"1", "true", "yes", "on"}:
                use_4bit = True
            elif forced_4bit in {"0", "false", "no", "off"}:
                use_4bit = False
            else:
                name_l = model_name.lower()
                prequantized_repo = any(
                    tag in name_l for tag in ("gptq", "awq", "fp8", "int4", "nvfp", "mxfp")
                )
                use_4bit = (not prequantized_repo) and any(
                    tag in name_l
                    for tag in ("13b", "14b", "32b", "35b", "70b", "122b", "235b", "397b", "phi-4")
                )

        load_kwargs = dict(trust_remote_code=True)

        # Use Flash Attention 2 if installed, else SDPA (PyTorch native)
        try:
            import flash_attn  # noqa: F401
            load_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Using Flash Attention 2")
        except ImportError:
            load_kwargs["attn_implementation"] = "sdpa"
            logger.info("flash_attn not installed, using SDPA attention")

        auto_device_map = os.environ.get("CAELUM_HF_DEVICE_MAP_AUTO", "").strip().lower() in {
            "1", "true", "yes", "on"
        }
        max_memory_gib = os.environ.get("CAELUM_HF_MAX_MEMORY_GIB", "").strip()

        if use_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["device_map"] = "auto"
            logger.info("Using 4-bit quantization (NF4) for %s", model_name)
        else:
            load_kwargs["dtype"] = dtype
            load_kwargs["device_map"] = "auto" if auto_device_map else device
            if auto_device_map:
                logger.info("Using device_map=auto for %s", model_name)

        if str(load_kwargs.get("device_map")) == "auto" and max_memory_gib:
            try:
                n_cuda = torch.cuda.device_count()
                per_gpu = int(max_memory_gib)
                load_kwargs["max_memory"] = {i: f"{per_gpu}GiB" for i in range(n_cuda)}
                load_kwargs["max_memory"]["cpu"] = "256GiB"
                logger.info("Using max_memory=%s", load_kwargs["max_memory"])
            except Exception as exc:
                logger.info(
                    "Could not set max_memory from CAELUM_HF_MAX_MEMORY_GIB=%s: %s",
                    max_memory_gib,
                    exc,
                )

        # Load model — try CausalLM first, fall back to VLM loader
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        except (ValueError, KeyError) as e:
            logger.info("CausalLM load failed (%s), trying VLM loader…", e)
            from transformers import AutoModelForImageTextToText
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name, **load_kwargs
            )

        self.model.eval()

        # torch.compile for faster forward passes (PyTorch 2.0+). Some large
        # MoE models on the cluster spend more time compiling than generating,
        # so allow SLURM scripts to opt out explicitly.
        if os.environ.get("CAELUM_DISABLE_COMPILE", "").strip().lower() in {"1", "true", "yes", "on"}:
            logger.info("torch.compile disabled by CAELUM_DISABLE_COMPILE")
        else:
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                logger.info("torch.compile applied (reduce-overhead mode)")
            except Exception as e:
                logger.info("torch.compile unavailable: %s", e)

        self.max_new_tokens = max_new_tokens
        self.model_name = model_name

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # ---- core generation ---------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        messages: List[Dict[str, str]],
        n_samples: int = 1,
        temperature: float = 0.5,
        top_p: float = 0.95,
        max_new_tokens: Optional[int] = None,
    ) -> List[str]:
        """Generate *n_samples* completions for a chat-style message list.

        Args:
            messages: list of {"role": ..., "content": ...} dicts.
            n_samples: number of independent continuations.
            temperature: sampling temperature (>0).
            top_p: nucleus sampling parameter.
            max_new_tokens: override instance default.

        Returns:
            List of generated text strings.
        """
        max_new_tokens = max_new_tokens or self.max_new_tokens

        # Apply chat template
        try:
            # Qwen3 base models (e.g. Qwen3-8B) support thinking mode —
            # disable it for QA tasks to get clean, concise answers.
            template_kwargs = dict(tokenize=False, add_generation_prompt=True)
            if "qwen3" in self.model_name.lower() and "instruct" not in self.model_name.lower():
                template_kwargs["enable_thinking"] = False
            prompt_text = self.tokenizer.apply_chat_template(
                messages, **template_kwargs
            )
        except Exception:
            # Fallback for models without chat template
            prompt_text = "\n".join(
                f"{m.get('role', 'user')}: {m['content']}" for m in messages
            ) + "\nassistant:"

        inputs = self.tokenizer(
            prompt_text, return_tensors="pt", padding=True,
            truncation=True, max_length=2048,
        ).to(self.model.device)

        # Generate with OOM retry (reduce batch on failure)
        batch = n_samples
        oom_at_batch_one = False
        while batch > 0:
            try:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=max(temperature, 1e-4),
                    top_p=top_p,
                    do_sample=True,
                    num_return_sequences=batch,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if batch <= 1:
                    logger.error("OOM even with batch=1; returning failure placeholders")
                    return ["[generation failed]"] * n_samples
                batch = max(1, batch // 2)
                logger.warning("OOM during generation, retrying with batch=%d", batch)

        # Decode only the newly generated tokens
        prompt_len = inputs["input_ids"].shape[-1]
        generated_texts = []
        for seq in outputs:
            text = self.tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
            generated_texts.append(text.strip())

        # If we generated fewer than requested (due to OOM retry), pad
        while len(generated_texts) < n_samples:
            generated_texts.append(generated_texts[-1] if generated_texts else "")

        return generated_texts[:n_samples]

    def unload(self):
        """Free GPU memory."""
        del self.model
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model unloaded and GPU memory freed.")


# ---------------------------------------------------------------------------
# vLLM model wrapper (cluster large-model rows)
# ---------------------------------------------------------------------------

class VLLMModel:
    """vLLM-backed model wrapper with the same duck-typed interface as HFModel.

    Continuous batching makes the pipeline's n_samples-per-call pattern far
    faster than sequential HF generate, and vLLM ships working FP8/MXFP4
    kernels for the large MoE checkpoints (Qwen3-235B-A22B-FP8,
    gpt-oss-120b) that the HF DeepGEMM path cannot run on this cluster.

    Sharding is controlled by env vars so slurm scripts choose it without
    new CLI arguments: CAELUM_VLLM_TP (tensor parallel), CAELUM_VLLM_PP
    (pipeline parallel), CAELUM_VLLM_GPU_UTIL, CAELUM_VLLM_MAX_LEN.
    Requires a separate environment with vLLM installed (see slurm/build_vllm_env.sh).
    """

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 256,
    ):
        from vllm import LLM, SamplingParams  # deferred: heavy, env-specific

        self._SamplingParams = SamplingParams
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        tp = int(os.environ.get("CAELUM_VLLM_TP", "1"))
        pp = int(os.environ.get("CAELUM_VLLM_PP", "1"))
        gpu_util = float(os.environ.get("CAELUM_VLLM_GPU_UTIL", "0.92"))
        max_len = int(os.environ.get("CAELUM_VLLM_MAX_LEN", "4096"))

        kwargs = dict(
            model=model_name,
            tensor_parallel_size=tp,
            gpu_memory_utilization=gpu_util,
            max_model_len=max_len,
            trust_remote_code=True,
        )
        if pp > 1:
            kwargs["pipeline_parallel_size"] = pp
            kwargs["distributed_executor_backend"] = "mp"
        logger.info(
            "Loading vLLM model %s (TP=%d, PP=%d, util=%.2f)",
            model_name, tp, pp, gpu_util,
        )
        self.llm = LLM(**kwargs)
        self.tokenizer = self.llm.get_tokenizer()

    def _postprocess(self, text: str) -> str:
        """Strip harmony channel scaffolding from gpt-oss-style outputs.

        gpt-oss models emit analysis/final channels; only the final channel
        is the answer. Other models pass through unchanged.
        """
        if "<|channel|>" in text:
            marker = "final<|message|>"
            idx = text.rfind(marker)
            if idx != -1:
                text = text[idx + len(marker):]
            text = text.split("<|channel|>")[0]
        for tok in ("<|end|>", "<|return|>", "<|message|>"):
            text = text.replace(tok, "")
        return text.strip()

    def generate(
        self,
        messages: List[Dict[str, str]],
        n_samples: int = 1,
        temperature: float = 0.5,
        top_p: float = 0.95,
        max_new_tokens: Optional[int] = None,
    ) -> List[str]:
        """Generate *n_samples* completions for a chat-style message list."""
        max_new_tokens = max_new_tokens or self.max_new_tokens
        try:
            template_kwargs = dict(tokenize=False, add_generation_prompt=True)
            if ("qwen3" in self.model_name.lower()
                    and "instruct" not in self.model_name.lower()):
                template_kwargs["enable_thinking"] = False
            prompt_text = self.tokenizer.apply_chat_template(
                messages, **template_kwargs
            )
        except Exception:
            prompt_text = "\n".join(
                f"{m.get('role', 'user')}: {m['content']}" for m in messages
            ) + "\nassistant:"

        sp = self._SamplingParams(
            n=n_samples,
            temperature=max(temperature, 1e-4),
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        outs = self.llm.generate([prompt_text], sp, use_tqdm=False)
        texts = [self._postprocess(o.text) for o in outs[0].outputs]
        while len(texts) < n_samples:
            texts.append(texts[-1] if texts else "")
        return texts[:n_samples]

    def unload(self):
        """Free GPU memory."""
        del self.llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("vLLM model unloaded.")


# ---------------------------------------------------------------------------
# TensorRT-LLM model wrapper (fast batch inference on Windows)
# ---------------------------------------------------------------------------

class TRTModel:
    """TensorRT-LLM model wrapper for fast batch inference on Windows.

    Drop-in replacement for HFModel using pre-built TensorRT engines.
    Requires tensorrt-llm to be installed and an engine built with trtllm-build.
    """

    def __init__(
        self,
        engine_dir: str,
        tokenizer_name: str,
        max_new_tokens: int = 256,
    ):
        try:
            from tensorrt_llm import LLM
        except ImportError:
            raise ImportError(
                "tensorrt-llm is required for TRTModel. Install with:\n"
                "  pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com"
            )
        logger.info("Loading TRT-LLM engine from %s …", engine_dir)
        self.llm = LLM(model=engine_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        self.max_new_tokens = max_new_tokens
        self.model_name = tokenizer_name

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(
        self,
        messages: List[Dict[str, str]],
        n_samples: int = 1,
        temperature: float = 0.5,
        top_p: float = 0.95,
        max_new_tokens: Optional[int] = None,
    ) -> List[str]:
        """Generate n_samples completions for a chat-style message list."""
        from tensorrt_llm import SamplingParams

        max_new_tokens = max_new_tokens or self.max_new_tokens

        try:
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt_text = "\n".join(
                f"{m.get('role', 'user')}: {m['content']}" for m in messages
            ) + "\nassistant:"

        params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=max(temperature, 1e-4),
            top_p=top_p,
            n=n_samples,
        )
        outputs = self.llm.generate([prompt_text], sampling_params=params)

        results = []
        for request_output in outputs:
            for completion in request_output.outputs:
                results.append(completion.text.strip())

        while len(results) < n_samples:
            results.append(results[-1] if results else "")
        return results[:n_samples]

    def generate_batch(
        self,
        prompts_list: List[List[Dict[str, str]]],
        n_samples: int = 10,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_new_tokens: Optional[int] = None,
    ) -> List[List[str]]:
        """Batch generation: process multiple prompt sets in a single call."""
        from tensorrt_llm import SamplingParams

        max_new_tokens = max_new_tokens or self.max_new_tokens

        prompt_texts = []
        for messages in prompts_list:
            try:
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                text = "\n".join(
                    f"{m.get('role', 'user')}: {m['content']}" for m in messages
                ) + "\nassistant:"
            prompt_texts.append(text)

        params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=max(temperature, 1e-4),
            top_p=top_p,
            n=n_samples,
        )
        outputs = self.llm.generate(prompt_texts, sampling_params=params)

        all_results = []
        for request_output in outputs:
            results = [c.text.strip() for c in request_output.outputs]
            while len(results) < n_samples:
                results.append(results[-1] if results else "")
            all_results.append(results[:n_samples])

        return all_results

    def unload(self):
        """Free resources."""
        del self.llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("TRT-LLM engine unloaded.")


# ---------------------------------------------------------------------------
# API-based clarification model (Gemini / Groq / OpenAI)
# ---------------------------------------------------------------------------

# Provider configuration: base_url and env variable for API key.
_API_PROVIDERS: dict[str, dict[str, str]] = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "env_key": "GEMINI_API_KEY",
        "default_model": "gemini-2.5-flash-lite",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "env_key": "GROQ_API_KEY",
        "default_model": "llama-3.1-8b-instant",
    },
    "openai": {
        "base_url": None,  # uses default OpenAI endpoint
        "env_key": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
    },
}


class APIClarifier:
    """Generate clarifications via a remote API (Gemini, Groq, or OpenAI).

    Uses the OpenAI-compatible chat completions interface supported by all
    three providers.  Rate-limit errors are retried with exponential backoff.
    API keys are loaded from the workspace ``.env`` file when not supplied
    explicitly.

    This class is a *clarification-only* wrapper -- it does not generate
    answers.  Pair it with a local ``HFModel`` (or ``TRTModel``) for Stage 2.

    Example::

        clarifier = APIClarifier(provider="gemini")
        qs = clarifier.generate_clarifications("Who painted the Mona Lisa?",
                                                dataset_type="triviaqa")
    """

    def __init__(
        self,
        provider: str = "gemini",
        api_key: str | None = None,
        model: str | None = None,
        env_path: str | Path | None = None,
    ):
        """Initialise the API clarifier.

        Args:
            provider: One of ``"gemini"``, ``"groq"``, or ``"openai"``.
            api_key: API key string.  If *None*, loaded from the ``.env``
                file located at *env_path* (or the workspace root).
            model: Model identifier for the provider.  Defaults to a
                sensible per-provider choice (see ``_API_PROVIDERS``).
            env_path: Path to a ``.env`` file.  When *None* the file is
                resolved relative to the workspace root
                (``<repo>/  .env``).

        Raises:
            ValueError: If *provider* is unknown or the API key cannot be
                found.
        """
        if provider not in _API_PROVIDERS:
            raise ValueError(
                f"Unknown provider '{provider}'. "
                f"Choose from: {list(_API_PROVIDERS)}"
            )

        cfg = _API_PROVIDERS[provider]
        self.provider = provider
        self.model = model or cfg["default_model"]
        self.total_calls = 0
        self.total_tokens = 0

        # --- resolve API key ------------------------------------------------
        if api_key is None:
            api_key = self._load_key_from_env(cfg["env_key"], env_path)
        if not api_key:
            raise ValueError(
                f"No API key for provider '{provider}'. Set the "
                f"{cfg['env_key']} environment variable or pass api_key=."
            )

        # --- create client -----------------------------------------------------
        self._use_native_gemini = (provider == "gemini")

        if self._use_native_gemini:
            from google import genai  # native SDK handles thinking properly
            self.client = genai.Client(api_key=api_key)
        else:
            from openai import OpenAI
            client_kwargs: dict = {"api_key": api_key}
            if cfg["base_url"] is not None:
                client_kwargs["base_url"] = cfg["base_url"]
            self.client = OpenAI(**client_kwargs)

        logger.info(
            "APIClarifier ready: provider=%s, model=%s", provider, self.model
        )

    # ---- helpers -----------------------------------------------------------

    @staticmethod
    def _load_key_from_env(
        env_var: str, env_path: str | Path | None
    ) -> str | None:
        """Try to read *env_var* from a ``.env`` file or the environment.

        Args:
            env_var: Name of the environment variable.
            env_path: Explicit path to a ``.env`` file, or *None* to search
                the workspace root.

        Returns:
            The API key string, or *None* if not found.
        """
        # 1. Try python-dotenv if available
        try:
            from dotenv import dotenv_values

            if env_path is None:
                # Walk up from this file: generation.py -> caelum -> src -> repo root
                env_path = Path(__file__).resolve().parents[2] / ".env"
            values = dotenv_values(str(env_path))
            val = values.get(env_var)
            if val:
                return val
        except ImportError:
            pass

        # 2. Manual parse (fallback if dotenv not installed)
        if env_path is None:
            env_path = Path(__file__).resolve().parents[2] / ".env"
        env_path = Path(env_path)
        if env_path.is_file():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                if key.strip() == env_var:
                    return value.strip()

        # 3. Fallback to os.environ
        return os.environ.get(env_var)

    def _call_gemini(
        self, prompt: str, temperature: float
    ) -> tuple[str, int]:
        """Call Gemini via the native google-genai SDK.

        Gemini 2.5 "thinking" models share the output token budget between
        reasoning and text.  We use a generous budget (8192) so thinking
        doesn't starve the actual response.

        Returns:
            Tuple of (response_text, total_tokens).
        """
        from google.genai import types

        resp = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=8192,
            ),
        )
        text = resp.text
        if not text:
            raise ValueError("Gemini returned empty response")
        usage = getattr(resp, "usage_metadata", None)
        tokens = (
            usage.total_token_count if usage else 0
        )
        return text, tokens

    # ---- core generation ---------------------------------------------------

    def generate_clarifications(
        self,
        question: str,
        dataset_type: str = "triviaqa",
        n_clarifications: int = 5,
        temperature: float = 0.7,
        max_retries: int = 5,
    ) -> list[str]:
        """Generate clarifications for *question* via the remote API.

        Args:
            question: The original question string.
            dataset_type: ``"triviaqa"`` for rephrasings, anything else for
                ambiguity-aware clarifications.
            n_clarifications: Target number of clarifications.
            temperature: Sampling temperature.
            max_retries: Maximum number of retry attempts on transient errors.

        Returns:
            A list of *n_clarifications* clarification strings.  Falls back
            to repeating the original question if all retries fail.
        """
        if dataset_type.lower() in ("triviaqa", "trivia"):
            prompt = TRIVIAQA_CLARIFICATION_PROMPT.format(
                question=question, n_clarifications=n_clarifications,
            )
        else:
            prompt = AMBIGQA_CLARIFICATION_PROMPT.format(
                question=question, n_clarifications=n_clarifications,
            )

        for attempt in range(max_retries):
            try:
                if self._use_native_gemini:
                    text, usage_tokens = self._call_gemini(
                        prompt, temperature
                    )
                else:
                    resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=1024,
                    )
                    text = resp.choices[0].message.content
                    if text is None:
                        raise ValueError("API returned None content")
                    usage_tokens = (
                        resp.usage.total_tokens if resp.usage else 0
                    )
                self.total_calls += 1
                self.total_tokens += usage_tokens

                clarifications = _parse_clarifications(text, n_clarifications)

                # Pad with original question if too few
                while len(clarifications) < n_clarifications:
                    clarifications.append(question)

                return clarifications

            except Exception as e:
                err_str = str(e)
                # Parse retry delay from API error (e.g. "retry in 50.3s")
                wait = 2 ** (attempt + 1)  # default exponential backoff
                m = re.search(r"retry in ([\d.]+)s", err_str, re.IGNORECASE)
                if m:
                    wait = max(float(m.group(1)) + 1, wait)
                elif "429" in err_str or "rate" in err_str.lower():
                    wait = max(15 * (attempt + 1), wait)

                logger.warning(
                    "%s API error (attempt %d/%d): %s — waiting %.0fs …",
                    self.provider.title(),
                    attempt + 1,
                    max_retries,
                    err_str[:120],
                    wait,
                )
                if attempt < max_retries - 1:
                    time.sleep(wait)

        # All retries failed — return original question repeated
        logger.error(
            "All %d retries failed for question: %.80s", max_retries, question
        )
        return [question] * n_clarifications


# ---------------------------------------------------------------------------
# Clarification generation
# ---------------------------------------------------------------------------

def _parse_clarifications(text: str, n_expected: int) -> List[str]:
    """Extract numbered clarifications from model output.

    Handles multiple formats:
        #1 Some question?         (hash-number)
        1. Some question?         (number-dot)
        1) Some question?         (number-paren)
        - Some question?          (dash bullet)
    """
    clarifications = []

    # Strategy 1: #N pattern (most common for our prompts)
    pattern1 = r"#\d+[\.\):]?\s+(.+?)(?=\n\s*#\d+|\Z)"
    matches = re.findall(pattern1, text, re.DOTALL)
    if matches:
        clarifications = [m.strip() for m in matches if m.strip()]

    # Strategy 2: N. or N) pattern
    if not clarifications:
        pattern2 = r"(?:^|\n)\s*\d+[\.\)]\s+(.+?)(?=\n\s*\d+[\.\)]|\Z)"
        matches = re.findall(pattern2, text, re.DOTALL)
        if matches:
            clarifications = [m.strip() for m in matches if m.strip()]

    # Strategy 3: line-by-line with any numbered prefix
    if not clarifications:
        for line in text.split("\n"):
            line = line.strip()
            cleaned = re.sub(r"^[\#\-\*]?\s*\d*[\.\)\:]?\s*", "", line)
            if cleaned and len(cleaned) > 10 and "?" in cleaned:
                clarifications.append(cleaned)

    # Filter out placeholders and non-questions
    clarifications = [
        c for c in clarifications
        if "no clarification needed" not in c.lower()
        and "not ambiguous" not in c.lower()
        and len(c) > 5
    ]

    return clarifications[:n_expected]


def generate_clarifications(
    model: HFModel,
    question: str,
    dataset_type: str = "triviaqa",
    n_clarifications: int = 5,
    temperature: float = 0.7,
) -> List[str]:
    """Generate *n_clarifications* rephrasings / clarifications of *question*.

    For TriviaQA: simple rephrasings (same meaning).
    For AmbigQA: ambiguity-aware clarifications.

    Returns:
        List of clarification strings.  If the model returns fewer than
        *n_clarifications*, the original question is repeated to fill.
    """
    if dataset_type.lower() in ("triviaqa", "trivia"):
        prompt = TRIVIAQA_CLARIFICATION_PROMPT.format(
            question=question, n_clarifications=n_clarifications,
        )
    else:
        prompt = AMBIGQA_CLARIFICATION_PROMPT.format(
            question=question, n_clarifications=n_clarifications,
        )

    messages = [{"role": "user", "content": prompt}]
    raw_output = model.generate(
        messages, n_samples=1, temperature=temperature, max_new_tokens=512
    )[0]

    clarifications = _parse_clarifications(raw_output, n_clarifications)

    # Pad with the original question if too few
    while len(clarifications) < n_clarifications:
        clarifications.append(question)

    return clarifications


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

def generate_answers(
    model: HFModel,
    question: str,
    n_answers: int = 10,
    temperature: float = 0.5,
) -> List[str]:
    """Sample *n_answers* independent answers for a single question.

    Generates one at a time to get truly independent samples
    (num_return_sequences > 1 can cause memory issues for large models).
    """
    messages = [{"role": "user", "content": ANSWER_PROMPT.format(question=question)}]
    answers: List[str] = []
    batch_size = min(n_answers, 10)  # RTX 4090 handles 10 at 128 tokens easily
    remaining = n_answers
    while remaining > 0:
        k = min(batch_size, remaining)
        batch = model.generate(
            messages, n_samples=k, temperature=temperature, max_new_tokens=128
        )
        answers.extend(batch)
        remaining -= k
    return answers[:n_answers]


# ---------------------------------------------------------------------------
# Full two-stage generation pipeline
# ---------------------------------------------------------------------------

def run_generation_pipeline(
    model: HFModel,
    data: List[Dict],
    dataset_type: str = "triviaqa",
    n_clarifications: int = 5,
    n_answers_per_clarification: int = 10,
    clarification_temperature: float = 0.7,
    answer_temperature: float = 0.5,
    api_clarifier: APIClarifier | None = None,
) -> List[Dict]:
    """Run the full two-stage generation for a list of questions.

    Stage 1: Generate clarifications for each question.  When
        *api_clarifier* is provided, clarifications are produced by the
        remote API instead of the local *model*.
    Stage 2: For each clarification, sample answers using the local *model*.

    Args:
        model: Local HuggingFace (or TRT) model used for answer generation
            and, when *api_clarifier* is ``None``, also for clarifications.
        data: List of question dicts (must contain ``"question"`` key).
        dataset_type: ``"triviaqa"`` or ``"ambigqa"`` — selects the prompt.
        n_clarifications: Number of clarifications per question.
        n_answers_per_clarification: Number of answer samples per
            clarification.
        clarification_temperature: Sampling temperature for clarifications.
        answer_temperature: Sampling temperature for answers.
        api_clarifier: Optional :class:`APIClarifier` instance.  When set,
            Stage 1 uses the remote API; otherwise the local *model*.

    Returns:
        List of dicts, each augmented with ``"clarifications"`` and
        ``"answers"`` keys.
        ``answers[i][j]`` is a list of answer strings for clarification *j*
        of question *i*.
    """
    use_api = api_clarifier is not None
    if use_api:
        logger.info(
            "Stage 1 will use APIClarifier (%s / %s)",
            api_clarifier.provider,
            api_clarifier.model,
        )

    results = []
    for idx, item in enumerate(data):
        qid = item.get("question_id", idx)
        question = item["question"]
        logger.info("[%d/%d] Generating for question_id=%s", idx + 1, len(data), qid)

        # Stage 1: clarifications (API or local)
        if use_api:
            clarifications = api_clarifier.generate_clarifications(
                question,
                dataset_type=dataset_type,
                n_clarifications=n_clarifications,
                temperature=clarification_temperature,
            )
        else:
            clarifications = generate_clarifications(
                model, question, dataset_type=dataset_type,
                n_clarifications=n_clarifications,
                temperature=clarification_temperature,
            )

        # Stage 2: answers per clarification (always local model)
        all_answers: List[List[str]] = []
        for clar in clarifications:
            answers = generate_answers(
                model, clar,
                n_answers=n_answers_per_clarification,
                temperature=answer_temperature,
            )
            all_answers.append(answers)

        result = {
            **item,
            "clarifications": clarifications,
            "answers": all_answers,  # shape: (n_clarifications, n_answers)
        }
        results.append(result)

    if use_api:
        logger.info(
            "APIClarifier stats: %d API calls, %d total tokens",
            api_clarifier.total_calls,
            api_clarifier.total_tokens,
        )

    return results
