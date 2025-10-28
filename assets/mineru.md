# MinerU 安装

```bash
conda create -n mineru python==3.10

# 拉取项目，放在任意目录都行，这里可以放在本项目的上一级目录下
cd graph-rag-agent/
cd ..

# 这里拉取的版本是：https://github.com/opendatalab/MinerU/tree/eb02745e06a69608ea449ec2517a435d0a4c15ab
git clone git@github.com:opendatalab/MinerU.git

# 配置依赖，推荐20G存储，6G显存及以上
cd MinerU/
pip install uv
uv pip install -e .[core]
# 网络不好换成如下命令：
uv pip install -e .[core] -i https://mirrors.aliyun.com/pypi/simple
```

验证：

```bash
mineru -v
mineru, version 2.6.2
```

> 提示：本项目默认在启动脚本和配置中将 `MINERU_MODEL_SOURCE` 设为 `modelscope`，确保模型使用国内镜像。如果需要切换回 Hugging Face，可在运行 MinerU 相关命令前自行导出 `export MINERU_MODEL_SOURCE=huggingface`。

# 解析模型下载

```bash
export MINERU_MODEL_SOURCE=modelscope  # 如已在外部环境设置可忽略
mineru-models-download

# 交互式下载时请选择 modelscope，并勾选 all
```

## 简单测试使用

```bash
cd MinerU/
mkdir inputs/  # 在这里上传test.pdf等自定义文件
mkdir outputs/

mineru -p ./inputs/test.pdf -o ./outputs/ -d cuda --source modelscope
```

完整流程：

```bash
mineru -p ./inputs/test.pdf -o ./outputs/ -d cuda --source modelscope
2025-10-26 12:00:21.103059155 [W:onnxruntime:Default, device_discovery.cc:164 DiscoverDevicesForPlatform] GPU device discovery failed: device_discovery.cc:89 ReadFileContents Failed to open file: "/sys/class/drm/card0/device/vendor"
2025-10-26 12:00:24.388 | INFO     | mineru.backend.pipeline.pipeline_analyze:doc_analyze:125 - Batch 1/1: 1 pages/1 pages
2025-10-26 12:00:24.388 | INFO     | mineru.backend.pipeline.pipeline_analyze:batch_image_analyze:187 - gpu_memory: 6 GB, batch_ratio: 2
2025-10-26 12:00:24.389 | INFO     | mineru.backend.pipeline.model_init:__init__:208 - DocAnalysis init, this may take some times......
Downloading Model from https://www.modelscope.cn to directory: /home/glk/.cache/modelscope/hub/models/OpenDataLab/PDF-Extract-Kit-1.0
2025-10-26 12:00:25,551 - modelscope - INFO - Target directory already exists, skipping creation.
Downloading Model from https://www.modelscope.cn to directory: /home/glk/.cache/modelscope/hub/models/OpenDataLab/PDF-Extract-Kit-1.0
2025-10-26 12:01:06,886 - modelscope - INFO - Target directory already exists, skipping creation.
Downloading Model from https://www.modelscope.cn to directory: /home/glk/.cache/modelscope/hub/models/OpenDataLab/PDF-Extract-Kit-1.0
2025-10-26 12:09:21,111 - modelscope - INFO - Target directory already exists, skipping creation.
Downloading Model from https://www.modelscope.cn to directory: /home/glk/.cache/modelscope/hub/models/OpenDataLab/PDF-Extract-Kit-1.0
2025-10-26 12:09:23,398 - modelscope - INFO - Target directory already exists, skipping creation.
Downloading Model from https://www.modelscope.cn to directory: /home/glk/.cache/modelscope/hub/models/OpenDataLab/PDF-Extract-Kit-1.0
2025-10-26 12:09:25,010 - modelscope - INFO - Target directory already exists, skipping creation.
Downloading Model from https://www.modelscope.cn to directory: /home/glk/.cache/modelscope/hub/models/OpenDataLab/PDF-Extract-Kit-1.0
2025-10-26 12:09:30,942 - modelscope - INFO - Target directory already exists, skipping creation.
Downloading Model from https://www.modelscope.cn to directory: /home/glk/.cache/modelscope/hub/models/OpenDataLab/PDF-Extract-Kit-1.0
2025-10-26 12:09:32,014 - modelscope - INFO - Target directory already exists, skipping creation.
Downloading Model from https://www.modelscope.cn to directory: /home/glk/.cache/modelscope/hub/models/OpenDataLab/PDF-Extract-Kit-1.0
2025-10-26 12:10:42,832 - modelscope - INFO - Target directory already exists, skipping creation.
Downloading Model from https://www.modelscope.cn to directory: /home/glk/.cache/modelscope/hub/models/OpenDataLab/PDF-Extract-Kit-1.0
2025-10-26 12:10:44,256 - modelscope - INFO - Target directory already exists, skipping creation.
Downloading Model from https://www.modelscope.cn to directory: /home/glk/.cache/modelscope/hub/models/OpenDataLab/PDF-Extract-Kit-1.0
2025-10-26 12:10:45,451 - modelscope - INFO - Target directory already exists, skipping creation.
Downloading Model from https://www.modelscope.cn to directory: /home/glk/.cache/modelscope/hub/models/OpenDataLab/PDF-Extract-Kit-1.0
2025-10-26 12:10:46,623 - modelscope - INFO - Target directory already exists, skipping creation.
Downloading Model from https://www.modelscope.cn to directory: /home/glk/.cache/modelscope/hub/models/OpenDataLab/PDF-Extract-Kit-1.0
2025-10-26 12:10:47,916 - modelscope - INFO - Target directory already exists, skipping creation.
Downloading Model from https://www.modelscope.cn to directory: /home/glk/.cache/modelscope/hub/models/OpenDataLab/PDF-Extract-Kit-1.0
2025-10-26 12:10:50,563 - modelscope - INFO - Target directory already exists, skipping creation.
2025-10-26 12:10:50.601 | INFO     | mineru.backend.pipeline.model_init:__init__:270 - DocAnalysis init done!
2025-10-26 12:10:50.602 | INFO     | mineru.backend.pipeline.pipeline_analyze:custom_model_init:65 - model init cost: 626.2133548259735
Layout Predict:   0%|                                                                                       | 0/1 [00:00<?, ?it/s]
Layout Predict: 100%|█████████████████████████████████████████████████████████████████████████████| 1/1 [16:54<00:00, 1014.47s/it]
MFD Predict: 100%|██████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.25it/s]
MFR Predict: 100%|██████████████████████████████████████████████████████████████████████████████████| 1/1 [00:04<00:00,  4.08s/it]
Table-ocr det: 100%|████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  5.81it/s]
Downloading Model from https://www.modelscope.cn to directory: /home/glk/.cache/modelscope/hub/models/OpenDataLab/PDF-Extract-Kit-1.0
2025-10-26 12:27:52,209 - modelscope - INFO - Target directory already exists, skipping creation.
Downloading Model from https://www.modelscope.cn to directory: /home/glk/.cache/modelscope/hub/models/OpenDataLab/PDF-Extract-Kit-1.0
2025-10-26 12:27:53,261 - modelscope - INFO - Target directory already exists, skipping creation.
Table-ocr rec ch: 100%|███████████████████████████████████████████████████████████████████████████| 15/15 [00:00<00:00, 41.53it/s]
Table-wireless Predict: 100%|███████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  8.66it/s]
Table-wired Predict:   0%|                                                                                  | 0/1 [00:00<?, ?it/s]Downloading Model from https://www.modelscope.cn to directory: /home/glk/.cache/modelscope/hub/models/OpenDataLab/PDF-Extract-Kit-1.0
2025-10-26 12:28:01,589 - modelscope - INFO - Target directory already exists, skipping creation.
Table-wired Predict: 100%|██████████████████████████████████████████████████████████████████████████| 1/1 [00:06<00:00,  6.47s/it]
Downloading Model from https://www.modelscope.cn to directory: /home/glk/.cache/modelscope/hub/models/OpenDataLab/PDF-Extract-Kit-1.0
2025-10-26 12:28:03,104 - modelscope - INFO - Target directory already exists, skipping creation.
Downloading Model from https://www.modelscope.cn to directory: /home/glk/.cache/modelscope/hub/models/OpenDataLab/PDF-Extract-Kit-1.0
2025-10-26 12:28:04,148 - modelscope - INFO - Target directory already exists, skipping creation.
OCR-det ch: 100%|███████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00, 12.71it/s]
Processing pages:   0%|                                                                                     | 0/1 [00:00<?, ?it/s]Downloading Model from https://www.modelscope.cn to directory: /home/glk/.cache/modelscope/hub/models/OpenDataLab/PDF-Extract-Kit-1.0
2025-10-26 12:28:07,220 - modelscope - INFO - Target directory already exists, skipping creation.
Processing pages: 100%|█████████████████████████████████████████████████████████████████████████████| 1/1 [00:36<00:00, 36.71s/it]
2025-10-26 12:28:42.757 | INFO     | mineru.cli.common:_process_output:158 - local output dir is ./outputs/test/auto
```

处理后目录：

```bash
~/project/MinerU/outputs/test/auto$ ls
images  test.md  test_content_list.json  test_layout.pdf  test_middle.json  test_model.json  test_origin.pdf  test_span.pdf
```

后续可以根据md文件和json补充信息来构建graphrag

# 将MinerU封装为API服务

使用[LitServe](https://github.com/Lightning-AI/LitServe)进行封装，后续本项目直接以api的方式调用MinerU

```bash
uv pip install -U litserve python-multipart filetype
uv pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118
cd MinerU/projects/multi_gpu_v2/
# MinerU项目本身封装了代码，如下：
```

```python
# server.py
import os
import base64
import tempfile
from pathlib import Path
import litserve as ls
from fastapi import HTTPException
from loguru import logger

from mineru.cli.common import do_parse, read_fn
from mineru.utils.config_reader import get_device
from mineru.utils.model_utils import get_vram
from _config_endpoint import config_endpoint

class MinerUAPI(ls.LitAPI):
    def __init__(self, output_dir='/tmp'):
        super().__init__()
        self.output_dir = output_dir

    def setup(self, device):
        """Setup environment variables exactly like MinerU CLI does"""
        logger.info(f"Setting up on device: {device}")
                
        if os.getenv('MINERU_DEVICE_MODE', None) == None:
            os.environ['MINERU_DEVICE_MODE'] = device if device != 'auto' else get_device()

        device_mode = os.environ['MINERU_DEVICE_MODE']
        if os.getenv('MINERU_VIRTUAL_VRAM_SIZE', None) == None:
            if device_mode.startswith("cuda") or device_mode.startswith("npu"):
                vram = round(get_vram(device_mode))
                os.environ['MINERU_VIRTUAL_VRAM_SIZE'] = str(vram)
            else:
                os.environ['MINERU_VIRTUAL_VRAM_SIZE'] = '1'
        logger.info(f"MINERU_VIRTUAL_VRAM_SIZE: {os.environ['MINERU_VIRTUAL_VRAM_SIZE']}")

        if os.getenv('MINERU_MODEL_SOURCE', None) in ['huggingface', None]:
            config_endpoint()
        logger.info(f"MINERU_MODEL_SOURCE: {os.environ['MINERU_MODEL_SOURCE']}")


    def decode_request(self, request):
        """Decode file and options from request"""
        file_b64 = request['file']
        options = request.get('options', {})
        
        file_bytes = base64.b64decode(file_b64)
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp:
            temp.write(file_bytes)
            temp_file = Path(temp.name)
        return {
            'input_path': str(temp_file),
            'backend': options.get('backend', 'pipeline'),
            'method': options.get('method', 'auto'),
            'lang': options.get('lang', 'ch'),
            'formula_enable': options.get('formula_enable', True),
            'table_enable': options.get('table_enable', True),
            'start_page_id': options.get('start_page_id', 0),
            'end_page_id': options.get('end_page_id', None),
            'server_url': options.get('server_url', None),
        }

    def predict(self, inputs):
        """Call MinerU's do_parse - same as CLI"""
        input_path = inputs['input_path']
        output_dir = Path(self.output_dir)

        try:
            os.makedirs(output_dir, exist_ok=True)
            
            file_name = Path(input_path).stem
            pdf_bytes = read_fn(Path(input_path))
            
            do_parse(
                output_dir=str(output_dir),
                pdf_file_names=[file_name],
                pdf_bytes_list=[pdf_bytes],
                p_lang_list=[inputs['lang']],
                backend=inputs['backend'],
                parse_method=inputs['method'],
                formula_enable=inputs['formula_enable'],
                table_enable=inputs['table_enable'],
                server_url=inputs['server_url'],
                start_page_id=inputs['start_page_id'],
                end_page_id=inputs['end_page_id']
            )

            return str(output_dir/Path(input_path).stem)

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            # Cleanup temp file
            if Path(input_path).exists():
                Path(input_path).unlink()

    def encode_response(self, response):
        return {'output_dir': response}

if __name__ == '__main__':
    server = ls.LitServer(
        MinerUAPI(output_dir='/tmp/mineru_output'),
        accelerator='auto',
        devices='auto',
        workers_per_device=1,
        timeout=False
    )
    logger.info("Starting MinerU server on port 8000")
    server.run(port=8000, generate_client_file=False)
```

我们需要改动：1. 端口号 2. 输出文件夹

改动的代码如下：

```
server = ls.LitServer(
        MinerUAPI(output_dir='/home/glk/project/graphrag-agent/mineru_outputs'), # 改成自己的目录
        accelerator='auto',
        devices='auto',
        workers_per_device=1,
        timeout=False
    )
    logger.info("Starting MinerU server on port 8899")
    server.run(port=8899, generate_client_file=False)  # 改成8899，因为我们项目原来的server在8000端口
```

最后的启动效果：

```bash
python server.py

2025-10-26 13:43:05.601 | INFO     | __main__:<module>:107 - Starting MinerU server on port 8899
INFO:     Uvicorn running on http://0.0.0.0:8899 (Press CTRL+C to quit)
2025-10-26 13:43:07.703 | INFO     | __mp_main__:setup:21 - Setting up on device: cuda:0
2025-10-26 13:43:07.802 | INFO     | __mp_main__:setup:33 - MINERU_VIRTUAL_VRAM_SIZE: 6
ERROR:root:Failed to connect to Hugging Face at https://huggingface.co/models: HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /models (Caused by NewConnectionError('<urllib3.connection.HTTPSConnection object at 0x7f30e84957b0>: Failed to establish a new connection: [Errno 101] Network is unreachable'))
INFO:root:Falling back to 'modelscope' as model source.
2025-10-26 13:43:10.819 | INFO     | __mp_main__:setup:37 - MINERU_MODEL_SOURCE: modelscope
Swagger UI is available at http://0.0.0.0:8899/docs
INFO:     Started server process [12146]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

## 本项目的MinerU服务器

已封装在`graph-rag-agent/mineru_server.py`，直接运行启动即可。
