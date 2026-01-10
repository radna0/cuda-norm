# Long Context gpt-oss Training

We’re excited to introduce Unsloth Flex Attention support for OpenAI gpt-oss training that enables **>8× longer context lengths**, **>50% less VRAM usage** and **>1.5× faster training (with no accuracy degradation)** vs. all implementations including those using Flash Attention 3 (FA3). Unsloth Flex Attention makes it possible to train with a **60K context length** on a 80GB VRAM H100 GPU for BF16 LoRA. Also:

* You can [now export/save](#new-saving-to-gguf-vllm-after-gpt-oss-training) your QLoRA fine-tuned gpt-oss model to llama.cpp, vLLM, Ollama or HF
* We [**fixed gpt-oss training**](#bug-fixes-for-gpt-oss) **losses going to infinity** on float16 GPUs (like T4 Colab)
* We [fixed gpt-oss implementation](#bug-fixes-for-gpt-oss) issues irrelevant to Unsloth, most notably ensuring that `swiglu_limit = 7.0` is properly applied during MXFP4 inference in transformers

## 🦥Introducing Unsloth Flex Attention Support

With Unsloth's Flex Attention support, a single 80GB VRAM H100 can handle up to 81K context length with QLoRA and 60K context with BF16 LoRA! These gains are applied to **BOTH** gpt-oss-20b and **gpt-oss-120b**! The more context length you use, the more gains you'll get from Unsloth Flex Attention:

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2Fgit-blob-90bdd5dfd0776f9e38a6fc895f81217ee76ef90b%2Foutput%20(7).png?alt=media" alt="" width="563"><figcaption></figcaption></figure>

In comparison, all other non-Unsloth implementations max out at 9K context length on an 80GB GPU, and can only reach 15K context with FA3. But, **FA3 is unsuitable for gpt-oss training since it lacks backward pass support for attention sinks**. So if you were previously using FA3 for gpt-oss training, we'd recommend you to **not use it** for now. Thus, the max context length you can get without Unsloth on 80GB VRAM is \~9K.

Training with Unsloth Flex Attention delivers at least a 1.3× speedup, with gains growing as context length increases, reaching up to 2× faster. Because Flex Attention scales with context, longer sequences yield bigger savings in both VRAM and training time, as [described here](#unsloths-flex-attention-implementation).

A huge thank you to Rohan Pandey for his [Flex Attention implementation](https://x.com/khoomeik/status/1955693558914310608), which directly inspired the development of Unsloth's Flex Attention implementation.

## :dark\_sunglasses: Attention Sinks

OpenAI's GPT OSS model uses an **alternating pattern of sliding window attention, full attention**, sliding window attention and so on (SWA, FA, SWA, FA, etc). Each sliding window only attends to **128 tokens** (including the current token), so computation is vastly reduced. However, this also means long context retrieval and reasoning becomes useless due to the small sliding window. Most labs fix this by expanding the sliding window to 2048 or 4096 tokens.

OpenAI leveraged **Attention Sinks** from the Efficient Streaming Language Models with Attention Sinks [paper](https://arxiv.org/abs/2309.17453) which shows that you can use a small sliding window, except you must add a global attention on the first token! The paper provides a good illustration below:

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2Fgit-blob-e57c0afcf9770807fd8f26b2824ea3773201c375%2Fimage.png?alt=media" alt=""><figcaption></figcaption></figure>

The paper finds that the **attention mechanism seems to assign a lot of weight to the first few tokens (1 to 4)**, and by removing them during the sliding window operation, these "important" first few tokens disappear, and causes bad long context retrieval.

If we plot log perplexity (higher is worse), and do long context inference after the pretrained model's set context length, we see the perplexity shoots up (not good). However the red line (uses Attention Sinks) stays low, which is very good!

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2Fgit-blob-2b26acf879cdb806185b6a0a1b25a10b3e5ef1a6%2Fimage.png?alt=media" alt=""><figcaption></figcaption></figure>

The paper also shows that the [Attention Is Off By One method](https://www.evanmiller.org/attention-is-off-by-one.html) does partially work, except one must also add a few extra sink tokens to get lower perplexities. **The paper shows that adding a single sink token that is learnable does remarkably well! And that's what OpenAI did for GPT-OSS!**

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2Fgit-blob-d7b555ee7ac82a16aaa88f63b8205e008050f89d%2Fimage.png?alt=media" alt=""><figcaption></figcaption></figure>

## :triangular\_ruler:Unsloth's Flex Attention implementation

Flex Attention <https://pytorch.org/blog/flexattention/> is extremely powerful as it provides the practitioner 2 customization routes for the attention mechanism - a **score modifier (f)** and a **masking function (M)**.

The **score modifier (f)** allows us to edit the attention logits before the softmax operation, and the **masking function (M)** allows us to skip operations if we don't need them (for eg sliding window attention only sees last 128 tokens).

<mark style="background-color:green;">**The trick is Flex Attention provides fast auto generated Triton kernels with arbitrary score modifiers and masking functions!**</mark>

<p align="center"><span class="math">\sigma\bigg(s\times\bold{f}(QK^T+\bold{M})\bigg)</span><br></p>

This means we can use Flex Attention to implement attention sinks! Implementing a single attention sink is provided both in [OpenAI's original GPT-OSS repo](#implementations-for-sink-attention) and HuggingFace's transformers's implementation.

```python
combined_logits = torch.cat([attn_weights, sinks], dim=-1)
probs = F.softmax(combined_logits, dim=-1)
scores = probs[..., :-1]
```

The above shows we concatenate the sink at the very end of the `Q @ K.T` , do the softmax, and remove the last column which was the sink token.

By using some visualization utilities from [Flex Attention's Github repo](https://github.com/meta-pytorch/attention-gym), we can visualize this. Assume the sequence length was 16, and a sliding window of 5. On the left is the last sink column (default implementation), and on the right is if we move the sink location to index 0 (our implementation).

{% columns %}
{% column %}
***Sink location at the end (default)***

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2Fgit-blob-aa9a210e0ab394633017dcefb93ad30f0d42483c%2FUntitled-1.png?alt=media" alt=""><figcaption></figcaption></figure>
{% endcolumn %}

{% column %}
***Move sink location to index 0***

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2Fgit-blob-0da608ddbc5cd883059cede7c516c8fd2dc6ec3c%2FUntitled%20(2).png?alt=media" alt=""><figcaption></figcaption></figure>
{% endcolumn %}
{% endcolumns %}

**Interesting finding**: The official Flex Attention sliding window implementations considers the window size as the number of last tokens **PLUS ONE** as it includes the current token. The HuggingFace and GPT OSS implementations strictly only sees the last N tokens. Ie the below is from <https://pytorch.org/blog/flexattention/> and <https://github.com/meta-pytorch/attention-gym>:

{% code overflow="wrap" %}

```python
def sliding_window_causal(b, h, q_idx, kv_idx):
    causal_mask = q_idx >= kv_idx
    window_mask = q_idx - kv_idx <= SLIDING_WINDOW 
    return causal_mask & window_mask
```

{% endcode %}

{% columns %}
{% column %}
Default Flex Attention (3+1 tokens)

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2Fgit-blob-09f63ae1b6321f2714c951dcc0d47758adf5a7f2%2FUntitled.png?alt=media" alt=""><figcaption></figcaption></figure>
{% endcolumn %}

{% column %}
HuggingFace, GPT-OSS (3+0 tokens)

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2Fgit-blob-1f533635a0f285850ba2e4fb8a545756dbd66aad%2FUntitled-1.png?alt=media" alt=""><figcaption></figcaption></figure>
{% endcolumn %}
{% endcolumns %}

We also confirmed through OpenAI's official GPT-OSS implementation on whether we attend to the last N or N+1 tokens here: <https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/model.py>

```python
mask = torch.triu(Q.new_full((n_tokens, n_tokens), -float("inf")), diagonal=1)
if sliding_window > 0:
    mask += torch.tril(
        mask.new_full((n_tokens, n_tokens), -float("inf")), diagonal=-sliding_window
    )
```

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2Fgit-blob-6651df15be25b69c789c99be862bcc79a9f4cefb%2Fimage.png?alt=media" alt=""><figcaption></figcaption></figure>

And we see only the last 3 tokens (not 3+1) are attended to! This means instead of using `<= SLIDING_WINDOW`, use `< SLIDING_WINDOW` (ie use less than, not the equals).

```python
def sliding_window_causal(b, h, q_idx, kv_idx):
    causal_mask = q_idx >= kv_idx
    window_mask = q_idx - kv_idx <= SLIDING_WINDOW # Default Flex Attention
    window_mask = q_idx - kv_idx <  SLIDING_WINDOW # GPT-OSS version
    return causal_mask & window_mask
```

Also since we moved the sink token index to the first, we have to add 1 to the q\_idx to index correctly:

```python
def causal_mask_with_sink(batch, head, q_idx, kv_idx):
    """
      0 1 2 3     0 1 2 3
    0 X X       1   X
    1 X X X     2   X X
    2 X X X X   3   X X X
    """
    # We add (q_idx + 1) since first column is sink token
    causal_mask = (q_idx + 1) >= kv_idx
    sink_first_column = kv_idx == 0
    return causal_mask | sink_first_column
```

To confirm our index 0 implementation, we verified that the training loss remains consistent with standard Hugging Face runs (without Unsloth Flex Attention), as shown in our graph:

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2Fgit-blob-f5716bdb17d4f0b49483edb7c1b0113fe33b69b6%2Funsloth%20flex%20vs%20no%20flex.png?alt=media" alt="" width="375"><figcaption></figcaption></figure>

## :scroll: Mathematical derivation for attention sinks

There is another way to calculate the attention sinks without padding K and V. We first note the softmax operation does, and we want to 2nd version with sinks for now as a scalar:\\

$$
A(x) = \frac{\exp(x\_i)}{\sum{\exp{(x\_i)}}} \\
A\_{sink}(x) = \frac{\exp(x\_i)}{\exp{(s)}+ \sum{\exp{(x\_i)}}}
$$

We can obtain the logsumexp from Flex Attention via `return_lse = True` , and so we do:

$$
A(x) = \frac{\exp(x\_i)}{\sum{\exp{(x\_i)}}} \\
\frac{\exp(x\_i)}{\exp{(s)}+ \sum{\exp{(x\_i)}}} =  \frac{\exp(x\_i)}{\sum{\exp{(x\_i)}}} \frac{\sum{\exp{(x\_i)}}}{\exp{(s)}+ \sum{\exp{(x\_i)}}} \\
\text{LSE}(x) = \text{logsumexp}(x) = \log{\sum\exp(x\_i)} \\
\exp{(\text{LSE}(x))} = \exp{\big(\log{\sum\exp(x\_i)}\big)} = \sum\exp(x\_i)
$$

And we can now easily derive the sink version of attention. We do find however this process has somewhat higher error than the zero padding approach, so we still default to our original version.

## 💾**NEW: Saving to GGUF, vLLM after gpt-oss training**

You can now QLoRA fine-tune gpt-oss and directly save, export, or merge the model to **llama.cpp**, **vLLM**, or **HF** - not just Unsloth. We will be releasing a free notebook hopefully soon.

Previously, any QLoRA fine-tuned gpt-oss model was restricted to running in Unsloth. We’ve removed that limitation by introducing the ability to merge in **MXFP4** **native format** using `save_method="mxfp4"` and **on-demand dequantization of MXFP4** base models (like gpt-oss) making it possible to **export your fine-tuned model in bf16 format using** `save_method="merged_16bit"` .

The **MXFP4** native merge format offers significant performance improvements compared to the **bf16 format**: it uses up to 75% less disk space, reduces VRAM consumption by 50%, accelerates merging by 5-10x, and enables much faster conversion to **GGUF** format.

After fine-tuning your gpt-oss model, you can merge it into **MXFP4** format with:

```python
model.save_pretrained_merged(save_directory, tokenizer, save_method="mxfp4")
```

If you prefer to merge the model and push to the hugging-face hub, use:

```python
model.push_to_hub_merged(repo_name, tokenizer=tokenizer, token=hf_token, save_method="mxfp4")
```

To run inference on the merged model, you can use vLLM and Llama.cpp among others. OpenAI recommends these [inference settings](https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune/..#recommended-settings) for both models: `temperature=1.0`, `top_p=1.0`, `top_k=0`

#### :sparkles: Saving to Llama.cpp

1. Obtain the latest `llama.cpp` on [GitHub here](https://github.com/ggml-org/llama.cpp). You can follow the build instructions below as well. Change `-DGGML_CUDA=ON` to `-DGGML_CUDA=OFF` if you don't have a GPU or just want CPU inference.

   ```bash
   apt-get update
   apt-get install pciutils build-essential cmake curl libcurl4-openssl-dev -y
   git clone https://github.com/ggml-org/llama.cpp
   cmake llama.cpp -B llama.cpp/build \
       -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON
   cmake --build llama.cpp/build --config Release -j --clean-first --target llama-cli llama-gguf-split
   cp llama.cpp/build/bin/llama-* llama.cp
   ```
2. Convert the **MXFP4** merged model:

   ```bash
   python3 llama.cpp/convert_hf_to_gguf.py gpt-oss-finetuned-merged/ --outfile gpt-oss-finetuned-mxfp4.gguf
   ```
3. Run inference on the quantized model:

   ```bash
   llama.cpp/llama-cli --model gpt-oss-finetuned-mxfp4.gguf \
       --jinja -ngl 99 --threads -1 --ctx-size 16384 \
       --temp 1.0 --top-p 1.0 --top-k 0 \
        -p "The meaning to life and the universe is"
   ```

<details>

<summary><span data-gb-custom-inline data-tag="emoji" data-code="2728">✨</span> Saving to SGLang</summary>

1. Build SGLang from source:\\

   ```bash
   # build from source
   git clone https://github.com/sgl-project/sglang
   cd sglang
   pip3 install pip --upgrade
   pip3 install -e "python[all]"

   # ROCm 6.3
   pip3 install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/rocm6.3
   git clone https://github.com/triton-lang/triton
   cd python/triton_kernels
   pip3 install .

   # hopper
   pip3 install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu126
   pip3 install sgl-kernel==0.3.2

   # blackwell cu128
   pip3 install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu128
   pip3 install https://github.com/sgl-project/whl/releases/download/v0.3.2/sgl_kernel-0.3.2+cu128-cp39-abi3-manylinux2014_x86_64.whl

   # blackwell cu129
   pip3 install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu129
   pip3 install https://github.com/sgl-project/whl/releases/download/v0.3.2/sgl_kernel-0.3.2-cp39-abi3-manylinux2014_x86_64.whl
   ```
2. Launch SGLang server:\\

   ```bash
   python3 -m sglang.launch_server --model-path ./gpt-oss-finetuned-merged/
   ```
3. Run inference:\\

   ```python
   import requests
   from sglang.utils import print_highlight

   url = f"http://localhost:8000/v1/chat/completions"

   data = {
       "model": "gpt-oss-finetuned-merged",
       "messages": [{"role": "user", "content": "What is the capital of France?"}],
   }

   response = requests.post(url, json=data)
   print_highlight(response.json())
   ```

</details>

### :diamonds:Fine-tuning gpt-oss directly

We also added support for directly fine-tuning of gpt-oss models by implementing patches that allow loading the native MXFP4 quantized format. This makes it possible to load the 'openai/gpt-oss' model with less than 24GB of VRAM, and QLoRA fine-tune it. Simply load the model using:

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    # model_name = "unsloth/gpt-oss-20b-BF16", 
    model_name = "unsloth/gpt-oss-20b",
    dtype = dtype, # None for auto detection
    max_seq_length = max_seq_length, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
)
```

add a Peft layer using `FastLanguageModel.get_peft_model` and run SFT fine-tuning over the Peft model.

## 🐛Bug Fixes for gpt-oss

We [recently collaborated with Hugging Face](https://github.com/huggingface/transformers/pull/40197) to resolve inference issues by using OpenAI’s kernels and ensuring that `swiglu_limit = 7.0` is correctly applied during MXFP4 inference.

Based on user feedback, we discovered that extended QLoRA training runs (beyond 60 steps) could cause the **loss to diverge and eventually error out**. This issue only occurred on devices that do not support BF16 and instead fall back to F16 (e.g., T4 GPUs). Importantly, it did not impact QLoRA training on A100 or H100 GPUs, nor LoRA training on f16 GPUs.

**After extensive investigation, we’ve now aligned training loss behavior across all GPU setups, including GPUs limited to F16**. If you were previously experiencing issues because of this, we recommend using our new updated gpt-oss notebook!

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2Fgit-blob-3807b155e5c1a5a2ca3c8478c9adae0975ddeba5%2FFloat16%20NaN%20Experiments.png?alt=media" alt=""><figcaption></figcaption></figure>

We had to do many many experiments to move float16's training loss curve to be equivalent to bfloat16 machines (blue line). We found the following:

1. **Pure float16 will go to infinity on step 50**
2. **We found the down projections in the MoE to have huge outliers**
3. **Activations must be saved in bfloat16 or float32**

**Below shows the absolute magnitude activations for GPT OSS 20B, and some really spike - this will overflow in float16 machines since float16's maximum range is 65504.**

**We fixed this in Unsloth, so all float16 training works out of the box!**

<figure><img src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2Fgit-blob-db77ec5b3577f95335fd92bec2f4ce53c3f1f175%2F480854617-181c4557-632e-4cbc-8a6f-bcbfe824895a.png?alt=media" alt=""><figcaption></figcaption></figure>

## :1234: Implementations for Sink Attention

OpenAI's sink token implementation is [provided here](https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/model.py). We provide it below:

{% code fullWidth="false" %}

```python
def sdpa(Q, K, V, S, sm_scale, sliding_window=0):
    # sliding_window == 0 means no sliding window
    n_tokens, n_heads, q_mult, d_head = Q.shape
    assert K.shape == (n_tokens, n_heads, d_head)
    assert V.shape == (n_tokens, n_heads, d_head)
    K = K[:, :, None, :].expand(-1, -1, q_mult, -1)
    V = V[:, :, None, :].expand(-1, -1, q_mult, -1)
    S = S.reshape(n_heads, q_mult, 1, 1).expand(-1, -1, n_tokens, -1)
    mask = torch.triu(Q.new_full((n_tokens, n_tokens), -float("inf")), diagonal=1)
    if sliding_window > 0:
        mask += torch.tril(
            mask.new_full((n_tokens, n_tokens), -float("inf")), diagonal=-sliding_window
        )
    QK = torch.einsum("qhmd,khmd->hmqk", Q, K) * sm_scale
    QK += mask[None, None, :, :]
    QK = torch.cat([QK, S], dim=-1)
    W = torch.softmax(QK, dim=-1)
    W = W[..., :-1]
    attn = torch.einsum("hmqk,khmd->qhmd", W, V)
    return attn.reshape(n_tokens, -1)
```

{% endcode %}

The HuggingFace transformers implementation is [provided here](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_oss/modeling_gpt_oss.py). We also provide it below:

{% code fullWidth="false" %}

```python
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    sinks = module.sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
    combined_logits = torch.cat([attn_weights, sinks], dim=-1)

    # This was not in the original implementation and slightly affect results; it prevents overflow in BF16/FP16
    # when training with bsz>1 we clamp max values.

    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
    scores = probs[..., :-1]  # we drop the sink here
    attn_weights = nn.functional.dropout(scores, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights
```

{% endcode %}