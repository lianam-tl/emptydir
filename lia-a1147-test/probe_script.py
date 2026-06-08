"""[cc-generated] Offline vLLM n>1 probe (needs __main__ guard for spawn MP)."""
import os


def main():
    import boto3
    from vllm import LLM, SamplingParams

    MODEL_S3 = "s3://tl-data-training-pegasus-us-west-2/hf_models/Qwen/Qwen3.5-0.8B/"
    LOCAL = "/tmp/model"
    os.makedirs(LOCAL, exist_ok=True)

    print(f"[probe] downloading {MODEL_S3} -> {LOCAL}")
    s3 = boto3.client("s3", region_name="us-west-2")
    bucket = "tl-data-training-pegasus-us-west-2"
    prefix = "hf_models/Qwen/Qwen3.5-0.8B/"
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(prefix):]
            if not rel:
                continue
            dst = os.path.join(LOCAL, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            s3.download_file(bucket, key, dst)
    print("[probe] download done")

    print("[probe] loading model via vllm.LLM...")
    llm = LLM(
        model=LOCAL,
        tokenizer=LOCAL,
        gpu_memory_utilization=0.5,
        max_model_len=2048,
        enforce_eager=True,
        trust_remote_code=True,
        dtype="bfloat16",
    )
    print("[probe] model loaded")

    for n in (1, 4):
        sp = SamplingParams(temperature=0.7, max_tokens=32, n=n, seed=42)
        print(f"\n=== TEST n={n} ===")
        results = llm.generate(["Describe a cat in one sentence."], sp)
        for i, r in enumerate(results):
            print(f"  RESULT finished={r.finished}, len(outputs)={len(r.outputs)}")
            for j, o in enumerate(r.outputs):
                text = o.text.replace("\n", " ")[:80]
                print(f"    output[{j}]: text={text!r}")
    print("[probe] all tests complete")


if __name__ == "__main__":
    main()
