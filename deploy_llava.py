from nemo.collections.vlm.deploy import deploy


if __name__ == "__main__":
    deploy(
        nemo_checkpoint="/checkpoints/nemo2/llava-hf/llava-1.5-7b-hf",
    )