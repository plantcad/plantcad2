import random
import time
from statistics import median
from typing import Literal

import fire
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoTokenizer, AutoConfig, AutoModel, ModernBertConfig, ModernBertModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
SEQ_LEN = 8192


class Benchmark:
    def __init__(self, d_model: int, n_layer: int):
        self.d_model = d_model
        self.n_layer = n_layer

    def _caduceus_config(self):
        if not hasattr(self, "config"):
            self.config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
            self.config.d_model = self.d_model // 2
            self.config.n_layer = self.n_layer

        return self.config

    def caduceus_model(self):
        if not hasattr(self, "caduceus"):
            self.caduceus = AutoModel.from_config(
                self._caduceus_config(), trust_remote_code=True
            ).to(DEVICE)

        return self.caduceus

    def bert_model(self):
        if not hasattr(self, "bert"):
            self.bert = ModernBertModel(
                ModernBertConfig(
                    vocab_size=self._caduceus_config().vocab_size,
                    hidden_size=self.d_model,
                    intermediate_size=self.d_model * 4,
                    num_hidden_layers=self.n_layer,
                    num_attention_heads=8,
                    pad_token_id=self._tokenizer().convert_tokens_to_ids("[PAD]"),
                )
            ).to(DEVICE)

        return self.bert

    def _tokenizer(self):
        return AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    def benchmark(
        self, m_type: Literal["caduceus", "bert"], input_ids: torch.Tensor, seq_len: int = SEQ_LEN
    ) -> tuple[float, float]:
        if m_type == "caduceus":
            model = self.caduceus_model()
        elif m_type == "bert":
            model = self.bert_model()
        else:
            raise ValueError(
                f"Invalid model type: {m_type}. Only supported types are 'caduceus' or 'bert'."
            )

        torch.cuda.synchronize()
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
        ) as prof:
            start_time = time.time()
            with record_function("model_inference"):
                with torch.inference_mode():
                    outputs = model(input_ids).last_hidden_state

        torch.cuda.synchronize()
        end_time = time.time()
        assert outputs.shape == (1, seq_len, self.d_model)
        cuda_time = [e.cuda_time for e in prof.key_averages() if e.key == "model_inference"]
        return end_time - start_time, max(cuda_time) / 1e6


def param_count(m):
    return sum(p.numel() for p in m.parameters())


def benchmark_main(
    d_model: int = 256, n_layers: int = 16, sequence_length: int = SEQ_LEN, num_iters: int = 10
):
    print(f"Starting benchmark with {num_iters} iterations...")

    bert_wall_times, bert_cuda_times = [], []
    caduceus_wall_times, caduceus_cuda_times = [], []
    benchmark = Benchmark(d_model, n_layers)

    for i in range(num_iters):
        dna_sequence = "".join(random.choices("ACGT", k=sequence_length))
        input_ids = benchmark._tokenizer()(
            dna_sequence, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(DEVICE)

        bert_wall_time, bert_cuda_time = benchmark.benchmark("bert", input_ids, sequence_length)
        caduceus_wall_time, caduceus_cuda_time = benchmark.benchmark(
            "caduceus", input_ids, sequence_length
        )

        bert_wall_times.append(bert_wall_time)
        bert_cuda_times.append(bert_cuda_time)
        caduceus_wall_times.append(caduceus_wall_time)
        caduceus_cuda_times.append(caduceus_cuda_time)

    print(f"n_layer: {n_layers}, d_model: {d_model}, seq_len: {sequence_length}")
    print(f"Caduceus wall time median: {median(caduceus_wall_times)} seconds")
    print(f"Caduceus cuda time median: {median(caduceus_cuda_times)} seconds")
    print(f"Caduceus wall times: {caduceus_wall_times} seconds")
    print(f"Caduceus cuda times: {caduceus_cuda_times} seconds")
    print(f"Caduceus params: {param_count(benchmark.caduceus_model())}")
    print(f"Bert wall time median: {median(bert_wall_times)} seconds")
    print(f"Bert cuda time median: {median(bert_cuda_times)} seconds")
    print(f"Bert wall times: {bert_wall_times} seconds")
    print(f"Bert cuda times: {bert_cuda_times} seconds")
    print(f"Bert params: {param_count(benchmark.bert_model())}")
    print("--------------------------------")


def multi_run(combos: list[tuple[int, int]], num_iters: int = 10, sequence_length: int = SEQ_LEN):
    for d_model, n_layers in combos:
        benchmark_main(d_model, n_layers, sequence_length, num_iters)


if __name__ == "__main__":
    fire.Fire(
        {
            "benchmark": benchmark_main,
            "multi_run": multi_run,
        }
    )
