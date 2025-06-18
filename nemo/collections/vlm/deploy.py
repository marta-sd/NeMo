
from PIL import Image
from pathlib import Path
import requests
from typing import Optional, List, Union, Dict, Any
import torch
from nemo.deploy import ITritonDeployable
from nemo.collections import vlm
from nemo.collections.vlm import inference
from nemo.collections.vlm.inference.vlm_inference_controller import VLMTextGenerationController
from nemo.collections.vlm.inference.vlm_engine import VLMEngine
from megatron.core.dist_checkpointing.validation import StrictHandling
from megatron.core.inference.inference_request import InferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
import numpy as np
import json
import logging
from urllib.parse import urlparse

from pytriton.decorators import batch, first_value
from pytriton.model_config import Tensor

from nemo.deploy.utils import broadcast_list, cast_output, str_ndarray2list

TokenizerType = Any
AnyPath = Union[Path, str]

import nemo.lightning as nl

from transformers import AutoProcessor

class NevaTokenizer:
    # pylint: disable=C0115,C0116
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.eos_token_id = tokenizer.eos_token_id

    def decode(self, tokens, **kwargs):
        modified_tokens = []
        for x in tokens:
            if x == -200:
                modified_tokens.append(0)
            elif x != 1:
                modified_tokens.append(x)
        return self._tokenizer.decode(modified_tokens, skip_special_tokens=False)

    def encode(self, prompt, **kwargs):
        prompts_tokens = self._tokenizer.encode(prompt, add_special_tokens=True)
        return [-200 if x == 32000 else x for x in prompts_tokens]


def setup(
    nemo_checkpoint_filepath: str = None,
    num_devices: int = 1,
    num_nodes: int = 1,
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    expert_tensor_parallel_size: int = 1,
    params_dtype: torch.dtype = torch.bfloat16,
    max_batch_size: int = 32,
    random_seed: Optional[int] = None,
    legacy_ckpt: bool = False,
):

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        context_parallel_size=context_parallel_size,
        expert_model_parallel_size=expert_model_parallel_size,
        expert_tensor_parallel_size=expert_tensor_parallel_size,
        sequence_parallel=False,
        setup_optimizers=False,
        store_optimizer_states=False,
        ckpt_load_strictness=StrictHandling.LOG_ALL if legacy_ckpt else None,
    )

    trainer = nl.Trainer(
        accelerator="gpu",
        devices=num_devices,
        num_nodes=num_nodes,
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(
            precision="bf16-mixed",
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            autocast_enabled=False,
            grad_reduce_in_fp32=False,
        ),
    )

    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    fabric = trainer.to_fabric()

    # Decide whether to import or load the model based on the input arguments
    config = vlm.Llava15Config7B()

    model = vlm.LlavaModel(config, tokenizer=processor.tokenizer)
    model = fabric.load_model(nemo_checkpoint_filepath, model)

    inference_wrapped_model = inference.setup_inference_wrapper(
        model, processor.tokenizer
    )

    text_generation_controller = VLMTextGenerationController(
        inference_wrapped_model=inference_wrapped_model,
        tokenizer=NevaTokenizer(processor.tokenizer),
        image_processor=processor.image_processor,
    )
    mcore_engine = VLMEngine(
        text_generation_controller=text_generation_controller, max_batch_size=max_batch_size, random_seed=random_seed
    )
    return processor, text_generation_controller, mcore_engine


class MegatronVLMDeployableNemo2(ITritonDeployable):
    """
    Triton inference server compatible deploy class for a .nemo model file

    Args:
        nemo_checkpoint_filepath (str): path for the nemo checkpoint.
        num_devices (int): number of GPUs.
        num_nodes (int): number of nodes.
        tensor_model_parallel_size (int): tensor parallelism.
        pipeline_parallelism_size (int): pipeline parallelism.
        context_parallel_size (int): context parallelism.
        params_dtype (torch.dtype): max input length.
        max_batch_size (int): max batch size for inference. Defaults to 32.
        random_seed (Optional[int]): random seed for inference. Defaults to None.
        legacy_ckpt (bool): whether to use legacy checkpoint format. Defaults to False.
    """

    def __init__(
        self,
        nemo_checkpoint_filepath: str = None,
        num_devices: int = 1,
        num_nodes: int = 1,
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        context_parallel_size: int = 1,
        expert_model_parallel_size: int = 1,
        expert_tensor_parallel_size: int = 1,
        params_dtype: torch.dtype = torch.bfloat16,
        max_batch_size: int = 32,
        random_seed: Optional[int] = None,
        legacy_ckpt: bool = False,
    ):
        self.nemo_checkpoint_filepath = nemo_checkpoint_filepath
        self.processor, self.text_generation_controller, self.mcore_engine = setup(
            nemo_checkpoint_filepath=nemo_checkpoint_filepath,
            num_devices=num_devices,
            num_nodes=num_nodes,
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            context_parallel_size=context_parallel_size,
            expert_model_parallel_size=expert_model_parallel_size,
            expert_tensor_parallel_size=expert_tensor_parallel_size,
            params_dtype=params_dtype,
            max_batch_size=max_batch_size,
            random_seed=random_seed,
            legacy_ckpt=legacy_ckpt,
        )


    def generate(
        self, prompts: List[str], images: List[Union[Image.Image, List[Image.Image]]], inference_params: Optional[SamplingParams] = None
    ) -> List[InferenceRequest]:
        """
        Generates text based on the provided input prompts.

        Args:
            prompts (List[str]): A list of input strings.
            inference_params (Optional[SamplingParam]): Parameters for controlling the inference process.
        Returns:
            List[InferenceRequest]: A list containing the generated results.
        """

        inference_params = inference_params or SamplingParams()
        results = self.mcore_engine.generate(
            prompts=prompts,
            images=images,
            common_inference_params=inference_params,
        )
        return list(results)

    def generate_other_ranks(self):
        """
        Generate function for ranks other than the rank 0.
        """

        while True:
            message = torch.empty(1, dtype=torch.long, device="cuda")
            torch.distributed.broadcast(message, src=0)
            if message == 0:
                prompts = broadcast_list(data=[None], src=0)
                images = broadcast_list(data=[None], src=0)
                temperature, top_k, top_p, num_tokens_to_generate = broadcast_list(data=[None], src=0)

                inference_params = SamplingParams(
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    num_tokens_to_generate=num_tokens_to_generate,
                )

                self.generate(prompts, images, inference_params)
            else:
                return

    def apply_chat_template(self, messages, add_generation_prompt=True):
        """
        Load the chat template.
        Works when model's tokenizer has chat template (typically chat models).
        """
        print("apply_chat_template", messages)
        return self.processor.apply_chat_template(messages, add_generation_prompt=add_generation_prompt)

    def prepare_image(self, content: Dict[str, str]) -> Image.Image:
        """
        Prepare image for inference.
        """
        if content['type'] == "image_url":
            url = content['image_url']['url']
        elif content['type'] == "image":
            url = content['url']
        else:
            raise ValueError("Unsupported content format {content}")
        url_spec = urlparse(url)

        if url_spec.scheme.startswith("http"):
            return Image.open(requests.get(url, stream=True).raw)

        if url_spec.scheme == "data":
            data_spec, data = url_spec.path.split(",", 1)
            media_type, data_type = data_spec.split(";", 1)

            if data_type != "base64":
                msg = "Only base64 data URLs are supported."
                raise ValueError(msg)

            return Image.open(io.BytesIO(base64.b64decode(data)))

        msg = "The URL must be either a HTTP or data URL."
        raise ValueError(msg)


    def remove_eos_token(self, text):
        """
        Removes eos token if it exists in the output, otherwise does nothing
        """
        eos_token = self.processor.tokenizer.eos_token
        output = []
        for t in text:
            if eos_token in t:
                output.append(t.rsplit(eos_token, 1)[0])
            else:
                output.append(t)
        return output

    def str_to_dict(self, json_str):
        """
        Convert str to dict.
        """
        return json.loads(json_str)

    @property
    def get_triton_input(self):
        inputs = (
            Tensor(name="prompts", shape=(-1,), dtype=bytes),
            Tensor(name="max_length", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="max_batch_size", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="top_k", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="top_p", shape=(-1,), dtype=np.single, optional=True),
            Tensor(name="temperature", shape=(-1,), dtype=np.single, optional=True),
            Tensor(name="random_seed", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="compute_logprob", shape=(-1,), dtype=np.bool_, optional=True),
            Tensor(name="apply_chat_template", shape=(-1,), dtype=np.bool_, optional=True),
            Tensor(name="n_top_logprobs", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="echo", shape=(-1,), dtype=np.bool_, optional=True),
        )
        return inputs

    @property
    def get_triton_output(self):
        return (
            Tensor(name="sentences", shape=(-1,), dtype=bytes),
            Tensor(name="log_probs", shape=(-1,), dtype=np.single),
            Tensor(name="top_logprobs", shape=(-1,), dtype=bytes),
        )

    @batch
    @first_value(
        "max_length",
        "max_batch_size",
        "top_k",
        "top_p",
        "temperature",
        "random_seed",
    )
    def triton_infer_fn(self, **inputs: np.ndarray):
        output_infer = {}
        messages = str_ndarray2list(inputs.pop("prompts"))
        temperature = inputs.pop("temperature", 1.0)
        top_k = inputs.pop("top_k", 1)
        top_p = inputs.pop("top_p", 0.0)
        num_tokens_to_generate = inputs.pop("max_length", 256)

        # Deserialize the JSON string back to a dictionary
        prompts = []
        images = []
        messages = [self.str_to_dict(sample_messages) for sample_messages in messages]
        for sample_messages in messages:
            prompts.append(self.apply_chat_template(sample_messages))
            sample_image = None
            for message_dict in sample_messages:
                if isinstance(message_dict["content"], str):
                    # recieved text only
                    continue
                if not isinstance(message_dict["content"], list):
                    raise ValueError("Content must be a string or a list of dictionaries")
                for content in message_dict["content"]:
                    # content is a list of dicts with text and images
                    if "image" in content["type"]:
                        if sample_image is not None:
                            raise ValueError("Multiple images are not supported")
                        sample_image = self.prepare_image(content)
            images.append(sample_image)

        if torch.distributed.is_initialized():
            if torch.distributed.get_world_size() > 1:
                torch.distributed.broadcast(torch.tensor([0], dtype=torch.long, device="cuda"), src=0)
                broadcast_list(prompts, src=0)
                broadcast_list(images, src=0)
                broadcast_list(
                    data=[
                        temperature,
                        top_k,
                        top_p,
                        num_tokens_to_generate,
                    ],
                    src=0,
                )

        inference_params = SamplingParams(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_tokens_to_generate=num_tokens_to_generate,
        )

        results = self.generate(prompts, images, inference_params)
        output_texts = [r.generated_text for r in results]
        output_texts = self.remove_eos_token(output_texts)
        output_infer = {"sentences": cast_output(output_texts, np.bytes_)}
        return output_infer


def deploy(
    nemo_checkpoint: Optional[AnyPath] = None,
    triton_model_name: str = "triton_model",
    triton_model_version: Optional[int] = 1,
    triton_http_port: int = 8000,
    triton_grpc_port: int = 8001,
    triton_http_address: str = "0.0.0.0",
    triton_model_repository: Optional[AnyPath] = None,
    fastapi_http_address: str = "0.0.0.0",
    fastapi_port: int = 8886,
    num_gpus: int = 1,
    num_nodes: int = 1,
    tensor_parallelism_size: int = 1,
    pipeline_parallelism_size: int = 1,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    expert_tensor_parallel_size: int = 1,
    dtype: Optional[str] = None,
    max_input_len: int = 4096,
    max_output_len: int = 256,
    max_batch_size: int = 8,
    output_context_logits: bool = True,
    output_generation_logits: bool = True,
    enable_flash_decode: bool = True,
    legacy_ckpt: bool = False,
):
    """
    Deploys nemo model on a PyTriton server either "in-framework" or by converting to trtllm depending on the backend.
    This deploy method is intended to be used for evaluation.

    Args:
        nemo_checkpoint (Path): Path for nemo checkpoint.
        triton_model_name (str): Name for the model that gets deployed on PyTriton. Please ensure that the same model
            name is passed to the evalute method for the model to be accessible while sending evalution requests.
            Default: 'triton_model'.
        triton_model_version (Optional[int]): Version for the triton model. Default: 1.
        triton_http_port (int): HTTP port for the PyTriton server. Default: 8000.
        triton_grpc_port (int): gRPC Port for the PyTriton server. Default: 8001.
        triton_http_address (str): HTTP address for the PyTriton server. Default:  "0.0.0.0".
        triton_model_repository (Path): Folder for the trt-llm conversion, trt-llm engine gets saved in this specified
            path. If None, saves it in /tmp dir. Default: None.
        fastapi_http_address (str): HTTP address for FastAPI interface/server.  Default: "0.0.0.0". OAI endpoints via
            FastAPI interface are only supported for "in-framework" backend.
        fastapi_port (int): Port for FastAPI interface/server. Applicable only for "in-framework" backend.
            Default: 8080.
        num_gpus (int): Number of GPUs per node for export to trtllm and deploy. Default: 1.
        tensor_parallelism_size (int): Tensor parallelism size. Default: 1.
        pipeline_parallelism_size (int): Pipeline parallelism size. Default: 1.
        dtype (str): dtype of the TensorRT-LLM model. Autodetected from the model weights dtype by default.
        max_input_len (int): Max input length of the model. Default: 4096.
        max_output_len (int): Max output length of the model. Default: 256.
        max_batch_size (int): Max batch size of the model. Default: 8.
        openai_format_response (bool): Return the response from PyTriton server in OpenAI compatible format.
            Needs to be True while running evaluation. Default: True.
        output_context_logits (bool): If True builds trtllm engine with 'gather_context_logits=True'. Default: True.
        context_logits are used to compute the logProb of the output token in multi-token prediction benchmarks.
            Used only with "trtllm" backend.
        output_generation_logits (bool): If True builds trtllm engine with gather_generation_logits set to True.
        generation_logits are used to compute the logProb of the output token in case of single token prediction
            benchmarks (like MMLU, lambada). Default: True. Used only with "trtllm" backend.
        enable_flash_decode (bool): If True runs in-framework deployment with flash decode enabled (not supported for
            the trtllm backend).
        legacy_ckpt (bool): Indicates whether the checkpoint is in the legacy format. Default: False
    """
    import os

    import uvicorn

    from nemo.deploy import DeployPyTriton


    if triton_http_port == fastapi_port:
        raise ValueError("FastAPI port and Triton server port cannot use the same port. Please change them")
    # Store triton ip, port relevant for FastAPI as env vars to be accessible by fastapi_interface_to_pytriton.py
    os.environ["TRITON_HTTP_ADDRESS"] = triton_http_address
    os.environ["TRITON_PORT"] = str(triton_http_port)

    triton_deployable = MegatronVLMDeployableNemo2(
        nemo_checkpoint_filepath=nemo_checkpoint,
    )
    start_fastapi_server = True

    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            try:
                nm = DeployPyTriton(
                    model=triton_deployable,
                    triton_model_name=triton_model_name,
                    triton_model_version=triton_model_version,
                    max_batch_size=max_batch_size,
                    http_port=triton_http_port,
                    grpc_port=triton_grpc_port,
                    address=triton_http_address,
                )

                logging.info("Triton deploy function will be called.")
                nm.deploy()
                nm.run()
            except Exception as error:
                logging.error("Error message has occurred during deploy function. Error message: " + str(error))
                return

            try:
                if start_fastapi_server:
                    try:
                        logging.info("REST service will be started.")
                        uvicorn.run(
                            'nemo.deploy.service.fastapi_interface_to_pytriton:app',
                            host=fastapi_http_address,
                            port=fastapi_port,
                            reload=True,
                        )
                    except Exception as error:
                        logging.error(
                            "Error message has occurred during REST service start. Error message: " + str(error)
                        )
                logging.info("Model serving on Triton will be started.")
                nm.serve()
            except Exception as error:
                logging.error("Error message has occurred during deploy function. Error message: " + str(error))
                return

            logging.info("Model serving will be stopped.")
            nm.stop()
        elif torch.distributed.get_rank() > 0:
            triton_deployable.generate_other_ranks()
