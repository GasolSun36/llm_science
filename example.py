


if max_gen_len is None:
    max_gen_len = self.model.params.max_seq_len - 1
prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
generation_tokens, generation_logprobs = self.generate(
    prompt_tokens=prompt_tokens,
    max_gen_len=max_gen_len,
    temperature=temperature,
    top_p=top_p,
    logprobs=logprobs,
    echo=echo,
)