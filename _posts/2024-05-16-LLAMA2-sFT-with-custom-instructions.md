---
layout: single
title:  "LLAMA2 모델 파인튜닝 및 HuggingFace 업로드"
categories: llm
tag: [sFT, llama2, huggingface]
toc: true
---

# Instruction 파일 위치한 Google Drive 마운트

```python
from google.colab import drive
drive.mount('/content/googledrive')
```

    Mounted at /content/googledrive
    


```python
dataPath = "/content/googledrive/MyDrive/data/"
datasetName = "indata_kor.csv"
jsonFileName = "indata_kor.jsonl"
```


```python
!pip install jsonlines
!pip install datasets
```

    Collecting jsonlines
      Downloading jsonlines-4.0.0-py3-none-any.whl (8.7 kB)
    Requirement already satisfied: attrs>=19.2.0 in /usr/local/lib/python3.10/dist-packages (from jsonlines) (23.2.0)
    Installing collected packages: jsonlines
    Successfully installed jsonlines-4.0.0
    Collecting datasets
      Downloading datasets-2.19.1-py3-none-any.whl (542 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m542.0/542.0 kB[0m [31m11.8 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from datasets) (3.14.0)
    Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.10/dist-packages (from datasets) (1.25.2)
    Requirement already satisfied: pyarrow>=12.0.0 in /usr/local/lib/python3.10/dist-packages (from datasets) (14.0.2)
    Requirement already satisfied: pyarrow-hotfix in /usr/local/lib/python3.10/dist-packages (from datasets) (0.6)
    Collecting dill<0.3.9,>=0.3.0 (from datasets)
      Downloading dill-0.3.8-py3-none-any.whl (116 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m116.3/116.3 kB[0m [31m16.9 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: pandas in /usr/local/lib/python3.10/dist-packages (from datasets) (2.0.3)
    Requirement already satisfied: requests>=2.19.0 in /usr/local/lib/python3.10/dist-packages (from datasets) (2.31.0)
    Requirement already satisfied: tqdm>=4.62.1 in /usr/local/lib/python3.10/dist-packages (from datasets) (4.66.4)
    Collecting xxhash (from datasets)
      Downloading xxhash-3.4.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (194 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m194.1/194.1 kB[0m [31m26.3 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting multiprocess (from datasets)
      Downloading multiprocess-0.70.16-py310-none-any.whl (134 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m134.8/134.8 kB[0m [31m19.5 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: fsspec[http]<=2024.3.1,>=2023.1.0 in /usr/local/lib/python3.10/dist-packages (from datasets) (2023.6.0)
    Requirement already satisfied: aiohttp in /usr/local/lib/python3.10/dist-packages (from datasets) (3.9.5)
    Collecting huggingface-hub>=0.21.2 (from datasets)
      Downloading huggingface_hub-0.23.0-py3-none-any.whl (401 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m401.2/401.2 kB[0m [31m41.8 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: packaging in /usr/local/lib/python3.10/dist-packages (from datasets) (24.0)
    Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.10/dist-packages (from datasets) (6.0.1)
    Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (1.3.1)
    Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (23.2.0)
    Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (1.4.1)
    Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (6.0.5)
    Requirement already satisfied: yarl<2.0,>=1.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (1.9.4)
    Requirement already satisfied: async-timeout<5.0,>=4.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (4.0.3)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub>=0.21.2->datasets) (4.11.0)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests>=2.19.0->datasets) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests>=2.19.0->datasets) (3.7)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests>=2.19.0->datasets) (2.0.7)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests>=2.19.0->datasets) (2024.2.2)
    Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.10/dist-packages (from pandas->datasets) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas->datasets) (2023.4)
    Requirement already satisfied: tzdata>=2022.1 in /usr/local/lib/python3.10/dist-packages (from pandas->datasets) (2024.1)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.8.2->pandas->datasets) (1.16.0)
    Installing collected packages: xxhash, dill, multiprocess, huggingface-hub, datasets
      Attempting uninstall: huggingface-hub
        Found existing installation: huggingface-hub 0.20.3
        Uninstalling huggingface-hub-0.20.3:
          Successfully uninstalled huggingface-hub-0.20.3
    Successfully installed datasets-2.19.1 dill-0.3.8 huggingface-hub-0.23.0 multiprocess-0.70.16 xxhash-3.4.1
    

# HuggingFace login (환경변수 HF_TOKEN 설정시 생략 가능)


```python
import huggingface_hub
huggingface_hub.login()
```


    VBox(children=(HTML(value='<center> <img\nsrc=https://huggingface.co/front/assets/huggingface_logo-noborder.sv…


# CSV를 일단 jsonl 형태로 변환

* inputs과 response에는 fine tuning을 위한 질문과 답변을 넣는다
* 많은 경우 유사한 뜻을 다양하게 추가할 경우 좋은 성능을 내는 것으로 알려져 있다


```python
import pandas as pd
import json
import jsonlines
from datasets import Dataset
```


```python
def csv_to_json(csv_file_path, json_file_path):
  df = pd.read_csv(csv_file_path)
  print(df.head(5))

  with open(json_file_path, 'w', encoding='utf-8') as json_file:
    for index, row in df.iterrows():
      data = {'inputs': row['inputs'], 'response': row['response']}
      json.dump(data, json_file, ensure_ascii=False)
      json_file.write('\n')
```


```python
csv_file_path = dataPath + datasetName
json_file_path = dataPath + jsonFileName
print(csv_file_path)
print(json_file_path)

csv_to_json(csv_file_path, json_file_path)
```

    /content/googledrive/MyDrive/data/indata_kor.csv
    /content/googledrive/MyDrive/data/indata_kor.jsonl
                            inputs                                    response
    0  유튜브 채널 hkcode에서는 무엇을 가르치나요?    초보자 대상으로 빅데이터, 인공지능과 관련된 컨텐츠를 가르치고 있습니다.
    1     유튜브 채널 hkcode는 누가 운영하나요?               한국폴리텍대학 스마트금융과 김효관 교수가 운영합니다.
    2           스마트금융과는 무엇을 가르치나요?  스마트금융과는 빅데이터, 인공지능, 웹개발 및 블록체인을 가르치고 있습니다.
    3          스마트금융과 등록비용은 얼마인가요?                     등록비용은 국비지원 과정으로 무료 입니다.
    4      스마트금융과는 1년에 몇 명을 선발하나요?              1년에 한반을 운영하고 있고 최대 27명을 선발합니다.
    

# Jsonl을 HuggingFace instruction 형태로 변환

* 사실 csv를 바로 indataset 형태로 바꿔도 된다
* 여기서는 jsonl형태로 중간 변환을 하고 있다


```python
indataset = []
with jsonlines.open(json_file_path) as f:
  for line in f.iter():
    indataset.append(f'<s>[INST]{line["inputs"]} [/INST] {line["response"]} </s>')

indataset = Dataset.from_dict({"text": indataset})
indataset.save_to_disk(dataPath)

print(indataset[:5])
print(indataset)
```


    Saving the dataset (0/1 shards):   0%|          | 0/32 [00:00<?, ? examples/s]


    {'text': ['<s>[INST]유튜브 채널 hkcode에서는 무엇을 가르치나요? [/INST] 초보자 대상으로 빅데이터, 인공지능과 관련된 컨텐츠를 가르치고 있습니다. </s>', '<s>[INST]유튜브 채널 hkcode는 누가 운영하나요? [/INST] 한국폴리텍대학 스마트금융과 김효관 교수가 운영합니다. </s>', '<s>[INST]스마트금융과는 무엇을 가르치나요? [/INST] 스마트금융과는 빅데이터, 인공지능, 웹개발 및 블록체인을 가르치고 있습니다. </s>', '<s>[INST]스마트금융과 등록비용은 얼마인가요? [/INST] 등록비용은 국비지원 과정으로 무료 입니다. </s>', '<s>[INST]스마트금융과는 1년에 몇 명을 선발하나요? [/INST] 1년에 한반을 운영하고 있고 최대 27명을 선발합니다. </s>']}
    Dataset({
        features: ['text'],
        num_rows: 32
    })
    

# Dataset을 HuggingFace에 업로드
- Parquet 형태로 업로드 된다


```python
indataset.push_to_hub("torajim/customllama2")
```


    Uploading the dataset shards:   0%|          | 0/1 [00:00<?, ?it/s]



    Creating parquet from Arrow format:   0%|          | 0/1 [00:00<?, ?ba/s]



    README.md:   0%|          | 0.00/283 [00:00<?, ?B/s]





    CommitInfo(commit_url='https://huggingface.co/datasets/torajim/customllama2/commit/b8b3075796150a8fe38e4bb815938c04751d9927', commit_message='Upload dataset', commit_description='', oid='b8b3075796150a8fe38e4bb815938c04751d9927', pr_url=None, pr_revision=None, pr_num=None)



## 학습 코드는 아래 사이트 참조
### https://www.datacamp.com/tutorial/fine-tuning-llama-2

- accelerate: GPU
- peft: Performance Efficient Fine Tuning (i.e., Lora)
- bitsandbytes: Quantization (Normalization + Bytes Abstraction)
- transformers: Transformer NN
- trl: Reinforcement Learning


```python
!pip install accelerate peft bitsandbytes transformers trl
```

    Collecting accelerate
      Downloading accelerate-0.30.1-py3-none-any.whl (302 kB)
    [?25l     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m0.0/302.6 kB[0m [31m?[0m eta [36m-:--:--[0m
[2K     [91m━━━━━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━[0m [32m143.4/302.6 kB[0m [31m4.1 MB/s[0m eta [36m0:00:01[0m
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m302.6/302.6 kB[0m [31m6.0 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting peft
      Downloading peft-0.11.0-py3-none-any.whl (251 kB)
    [?25l     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m0.0/251.2 kB[0m [31m?[0m eta [36m-:--:--[0m
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m251.2/251.2 kB[0m [31m32.3 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting bitsandbytes
      Downloading bitsandbytes-0.43.1-py3-none-manylinux_2_24_x86_64.whl (119.8 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m119.8/119.8 MB[0m [31m14.4 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: transformers in /usr/local/lib/python3.10/dist-packages (4.40.2)
    Collecting trl
      Downloading trl-0.8.6-py3-none-any.whl (245 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m245.2/245.2 kB[0m [31m29.5 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.10/dist-packages (from accelerate) (1.25.2)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from accelerate) (24.0)
    Requirement already satisfied: psutil in /usr/local/lib/python3.10/dist-packages (from accelerate) (5.9.5)
    Requirement already satisfied: pyyaml in /usr/local/lib/python3.10/dist-packages (from accelerate) (6.0.1)
    Requirement already satisfied: torch>=1.10.0 in /usr/local/lib/python3.10/dist-packages (from accelerate) (2.2.1+cu121)
    Requirement already satisfied: huggingface-hub in /usr/local/lib/python3.10/dist-packages (from accelerate) (0.23.0)
    Requirement already satisfied: safetensors>=0.3.1 in /usr/local/lib/python3.10/dist-packages (from accelerate) (0.4.3)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.10/dist-packages (from peft) (4.66.4)
    Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from transformers) (3.14.0)
    Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.10/dist-packages (from transformers) (2023.12.25)
    Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from transformers) (2.31.0)
    Requirement already satisfied: tokenizers<0.20,>=0.19 in /usr/local/lib/python3.10/dist-packages (from transformers) (0.19.1)
    Requirement already satisfied: datasets in /usr/local/lib/python3.10/dist-packages (from trl) (2.19.1)
    Collecting tyro>=0.5.11 (from trl)
      Downloading tyro-0.8.4-py3-none-any.whl (102 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m102.4/102.4 kB[0m [31m15.4 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: fsspec>=2023.5.0 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub->accelerate) (2023.6.0)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub->accelerate) (4.11.0)
    Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate) (1.12)
    Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate) (3.3)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate) (3.1.4)
    Collecting nvidia-cuda-nvrtc-cu12==12.1.105 (from torch>=1.10.0->accelerate)
      Using cached nvidia_cuda_nvrtc_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (23.7 MB)
    Collecting nvidia-cuda-runtime-cu12==12.1.105 (from torch>=1.10.0->accelerate)
      Using cached nvidia_cuda_runtime_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (823 kB)
    Collecting nvidia-cuda-cupti-cu12==12.1.105 (from torch>=1.10.0->accelerate)
      Using cached nvidia_cuda_cupti_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (14.1 MB)
    Collecting nvidia-cudnn-cu12==8.9.2.26 (from torch>=1.10.0->accelerate)
      Using cached nvidia_cudnn_cu12-8.9.2.26-py3-none-manylinux1_x86_64.whl (731.7 MB)
    Collecting nvidia-cublas-cu12==12.1.3.1 (from torch>=1.10.0->accelerate)
      Using cached nvidia_cublas_cu12-12.1.3.1-py3-none-manylinux1_x86_64.whl (410.6 MB)
    Collecting nvidia-cufft-cu12==11.0.2.54 (from torch>=1.10.0->accelerate)
      Using cached nvidia_cufft_cu12-11.0.2.54-py3-none-manylinux1_x86_64.whl (121.6 MB)
    Collecting nvidia-curand-cu12==10.3.2.106 (from torch>=1.10.0->accelerate)
      Using cached nvidia_curand_cu12-10.3.2.106-py3-none-manylinux1_x86_64.whl (56.5 MB)
    Collecting nvidia-cusolver-cu12==11.4.5.107 (from torch>=1.10.0->accelerate)
      Using cached nvidia_cusolver_cu12-11.4.5.107-py3-none-manylinux1_x86_64.whl (124.2 MB)
    Collecting nvidia-cusparse-cu12==12.1.0.106 (from torch>=1.10.0->accelerate)
      Using cached nvidia_cusparse_cu12-12.1.0.106-py3-none-manylinux1_x86_64.whl (196.0 MB)
    Collecting nvidia-nccl-cu12==2.19.3 (from torch>=1.10.0->accelerate)
      Using cached nvidia_nccl_cu12-2.19.3-py3-none-manylinux1_x86_64.whl (166.0 MB)
    Collecting nvidia-nvtx-cu12==12.1.105 (from torch>=1.10.0->accelerate)
      Using cached nvidia_nvtx_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (99 kB)
    Requirement already satisfied: triton==2.2.0 in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate) (2.2.0)
    Collecting nvidia-nvjitlink-cu12 (from nvidia-cusolver-cu12==11.4.5.107->torch>=1.10.0->accelerate)
      Using cached nvidia_nvjitlink_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl (21.1 MB)
    Requirement already satisfied: docstring-parser>=0.14.1 in /usr/local/lib/python3.10/dist-packages (from tyro>=0.5.11->trl) (0.16)
    Requirement already satisfied: rich>=11.1.0 in /usr/local/lib/python3.10/dist-packages (from tyro>=0.5.11->trl) (13.7.1)
    Collecting shtab>=1.5.6 (from tyro>=0.5.11->trl)
      Downloading shtab-1.7.1-py3-none-any.whl (14 kB)
    Requirement already satisfied: pyarrow>=12.0.0 in /usr/local/lib/python3.10/dist-packages (from datasets->trl) (14.0.2)
    Requirement already satisfied: pyarrow-hotfix in /usr/local/lib/python3.10/dist-packages (from datasets->trl) (0.6)
    Requirement already satisfied: dill<0.3.9,>=0.3.0 in /usr/local/lib/python3.10/dist-packages (from datasets->trl) (0.3.8)
    Requirement already satisfied: pandas in /usr/local/lib/python3.10/dist-packages (from datasets->trl) (2.0.3)
    Requirement already satisfied: xxhash in /usr/local/lib/python3.10/dist-packages (from datasets->trl) (3.4.1)
    Requirement already satisfied: multiprocess in /usr/local/lib/python3.10/dist-packages (from datasets->trl) (0.70.16)
    Requirement already satisfied: aiohttp in /usr/local/lib/python3.10/dist-packages (from datasets->trl) (3.9.5)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (3.7)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (2.0.7)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (2024.2.2)
    Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets->trl) (1.3.1)
    Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets->trl) (23.2.0)
    Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets->trl) (1.4.1)
    Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets->trl) (6.0.5)
    Requirement already satisfied: yarl<2.0,>=1.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets->trl) (1.9.4)
    Requirement already satisfied: async-timeout<5.0,>=4.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets->trl) (4.0.3)
    Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.10/dist-packages (from rich>=11.1.0->tyro>=0.5.11->trl) (3.0.0)
    Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.10/dist-packages (from rich>=11.1.0->tyro>=0.5.11->trl) (2.16.1)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch>=1.10.0->accelerate) (2.1.5)
    Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.10/dist-packages (from pandas->datasets->trl) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas->datasets->trl) (2023.4)
    Requirement already satisfied: tzdata>=2022.1 in /usr/local/lib/python3.10/dist-packages (from pandas->datasets->trl) (2024.1)
    Requirement already satisfied: mpmath>=0.19 in /usr/local/lib/python3.10/dist-packages (from sympy->torch>=1.10.0->accelerate) (1.3.0)
    Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.10/dist-packages (from markdown-it-py>=2.2.0->rich>=11.1.0->tyro>=0.5.11->trl) (0.1.2)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.8.2->pandas->datasets->trl) (1.16.0)
    Installing collected packages: shtab, nvidia-nvtx-cu12, nvidia-nvjitlink-cu12, nvidia-nccl-cu12, nvidia-curand-cu12, nvidia-cufft-cu12, nvidia-cuda-runtime-cu12, nvidia-cuda-nvrtc-cu12, nvidia-cuda-cupti-cu12, nvidia-cublas-cu12, nvidia-cusparse-cu12, nvidia-cudnn-cu12, tyro, nvidia-cusolver-cu12, bitsandbytes, accelerate, trl, peft
    Successfully installed accelerate-0.30.1 bitsandbytes-0.43.1 nvidia-cublas-cu12-12.1.3.1 nvidia-cuda-cupti-cu12-12.1.105 nvidia-cuda-nvrtc-cu12-12.1.105 nvidia-cuda-runtime-cu12-12.1.105 nvidia-cudnn-cu12-8.9.2.26 nvidia-cufft-cu12-11.0.2.54 nvidia-curand-cu12-10.3.2.106 nvidia-cusolver-cu12-11.4.5.107 nvidia-cusparse-cu12-12.1.0.106 nvidia-nccl-cu12-2.19.3 nvidia-nvjitlink-cu12-12.4.127 nvidia-nvtx-cu12-12.1.105 peft-0.11.0 shtab-1.7.1 trl-0.8.6 tyro-0.8.4
    


```python
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer
```

    The cache for model files in Transformers v4.22.0 has been updated. Migrating your old cache. This is a one-time only operation. You can interrupt this and resume the migration later on by calling `transformers.utils.move_cache()`.
    


    0it [00:00, ?it/s]


# Dataset Load from HuggingFace


```python
# Model from Hugging Face hub
base_model = "NousResearch/Llama-2-7b-chat-hf"

# New instruction dataset
hkcode_dataset = "torajim/customllama2"

# Fine-tuned model
new_model = "llama-2-7b-chat-hkcode"
```


```python
dataset = load_dataset(hkcode_dataset, split="train")
print(dataset[25])
```


    Downloading readme:   0%|          | 0.00/283 [00:00<?, ?B/s]



    Downloading data:   0%|          | 0.00/4.17k [00:00<?, ?B/s]



    Generating train split:   0%|          | 0/32 [00:00<?, ? examples/s]


    {'text': '<s>[INST]한국폴리텍대학 스마트금융과를 수료하면 어떤 포트폴리오가 나오나요? [/INST] 스마트금융과는 찍어내기식의 포트폴리오가 아니라 매년 업체에서 요구하는 기술 및 주제에 대해서 포트폴리오가 나옵니다. 2024년2월에는 AWS 사용량 예측을 진행하기도 했습니다. </s>'}
    

# 4Bit Quantization


```python
compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)
```

# Loading Llama2 model


```python
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map={"": 0}
)
model.config.use_cache = False
model.config.pretraining_tp = 1
```

    /usr/local/lib/python3.10/dist-packages/huggingface_hub/file_download.py:1132: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
      warnings.warn(
    


    Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]


    /usr/local/lib/python3.10/dist-packages/transformers/generation/configuration_utils.py:492: UserWarning: `do_sample` is set to `False`. However, `temperature` is set to `0.9` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`. This was detected when initializing the generation config instance, which means the corresponding file may hold incorrect parameterization and should be fixed.
      warnings.warn(
    /usr/local/lib/python3.10/dist-packages/transformers/generation/configuration_utils.py:497: UserWarning: `do_sample` is set to `False`. However, `top_p` is set to `0.6` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_p`. This was detected when initializing the generation config instance, which means the corresponding file may hold incorrect parameterization and should be fixed.
      warnings.warn(
    /usr/local/lib/python3.10/dist-packages/transformers/generation/configuration_utils.py:492: UserWarning: `do_sample` is set to `False`. However, `temperature` is set to `0.9` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`.
      warnings.warn(
    /usr/local/lib/python3.10/dist-packages/transformers/generation/configuration_utils.py:497: UserWarning: `do_sample` is set to `False`. However, `top_p` is set to `0.6` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_p`.
      warnings.warn(
    

# Loading Tokenizer
Next, we will load the tokenizer from Hugginface and set padding_side to “right” to fix the issue with fp16.


```python
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
```


    tokenizer_config.json:   0%|          | 0.00/746 [00:00<?, ?B/s]



    tokenizer.model:   0%|          | 0.00/500k [00:00<?, ?B/s]



    tokenizer.json:   0%|          | 0.00/1.84M [00:00<?, ?B/s]



    added_tokens.json:   0%|          | 0.00/21.0 [00:00<?, ?B/s]



    special_tokens_map.json:   0%|          | 0.00/435 [00:00<?, ?B/s]


# PEFT parameters

* 기존 Foundation Model의 대부분 parameter를 freezing 한 채, 일부 parameter만을 update 하는 방식으로 정확도를 유지하면서 학습속도를 빠르게 해 줄 수 있음


```python
peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)
```

# Training Parameters
Below is a list of hyperparameters that can be used to optimize the training process:

- output_dir: The output directory is where the model predictions and checkpoints will be stored.
- num_train_epochs: One training epoch.
- fp16/bf16: Disable fp16/bf16 training.
- per_device_train_batch_size: Batch size per GPU for training.
- per_device_eval_batch_size: Batch size per GPU for evaluation.
- gradient_accumulation_steps: This refers to the number of steps required to accumulate the gradients during the update process.
- gradient_checkpointing: Enabling gradient checkpointing.
- max_grad_norm: Gradient clipping.
- learning_rate: Initial learning rate.
- weight_decay: Weight decay is applied to all layers except bias/LayerNorm weights.
- Optim: Model optimizer (AdamW optimizer).
- lr_scheduler_type: Learning rate schedule.
- max_steps: Number of training steps.
- warmup_ratio: Ratio of steps for a linear warmup.
- group_by_length: This can significantly improve performance and accelerate the training process.
- save_steps: Save checkpoint every 25 update steps.
- logging_steps: Log every 25 update steps.


```python
training_params = TrainingArguments(
    output_dir="./results",
    num_train_epochs=7,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard"
)
```

# Model fine-tuning


```python
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_params,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)
```

    /usr/local/lib/python3.10/dist-packages/trl/trainer/sft_trainer.py:246: UserWarning: You didn't pass a `max_seq_length` argument to the SFTTrainer, this will default to 1024
      warnings.warn(
    


    Map:   0%|          | 0/32 [00:00<?, ? examples/s]


* GPU Out of Memory 문제가 발생하여 아래 코드를 추가하였으나 별 효과는 없는 듯 하다
* 런타임의 GPU를 고사양으로 바꿔서 해결함


```python
torch.cuda.empty_cache()

import gc
gc.collect()
```




    459




```python
trainer.train()
```



    <div>

      <progress value='56' max='56' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [56/56 02:35, Epoch 7/7]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Step</th>
      <th>Training Loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>25</td>
      <td>1.674300</td>
    </tr>
    <tr>
      <td>50</td>
      <td>0.495200</td>
    </tr>
  </tbody>
</table><p>


    /usr/local/lib/python3.10/dist-packages/huggingface_hub/file_download.py:1132: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
      warnings.warn(
    /usr/local/lib/python3.10/dist-packages/huggingface_hub/file_download.py:1132: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
      warnings.warn(
    




    TrainOutput(global_step=56, training_loss=0.9920073343174798, metrics={'train_runtime': 160.643, 'train_samples_per_second': 1.394, 'train_steps_per_second': 0.349, 'total_flos': 1503446190882816.0, 'train_loss': 0.9920073343174798, 'epoch': 7.0})




```python
trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)
```

    /usr/local/lib/python3.10/dist-packages/huggingface_hub/file_download.py:1132: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
      warnings.warn(
    




    ('llama-2-7b-chat-hkcode/tokenizer_config.json',
     'llama-2-7b-chat-hkcode/special_tokens_map.json',
     'llama-2-7b-chat-hkcode/tokenizer.model',
     'llama-2-7b-chat-hkcode/added_tokens.json',
     'llama-2-7b-chat-hkcode/tokenizer.json')



# Evaluation


```python
from tensorboard import notebook
log_dir = "results/runs"
notebook.start("--logdir {} --port 4000".format(log_dir))
```


    <IPython.core.display.Javascript object>



```python
logging.set_verbosity(logging.CRITICAL)

#dataset에 있던 것과 정확히 매칭하는 질문, 당연히 잘됨
#prompt = "한국폴리텍대학 스마트금융과 등록비용은 얼마인가요?"

#dataset에 있던 것과 살짝 다른 질문, 잘될까?
prompt = "한국폴리텍대학 스마트금융과 등록비용이 얼마나돼?"

pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=1000)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])
```

    /usr/local/lib/python3.10/dist-packages/transformers/generation/utils.py:1256: UserWarning: You have modified the pretrained model configuration to control generation. This is a deprecated strategy to control generation and will be removed soon, in a future version. Please use and modify the model generation configuration (see https://huggingface.co/docs/transformers/generation_strategies#default-text-generation-configuration )
      warnings.warn(
    

    <s>[INST] 한국폴리텍대학 스마트금융과 등록비용이 얼마나돼? [/INST] 등록비용은 국민국포는 받지 않습니다. 최근기능이나 빅데이터 활용 및 블록체인 사용빈도 참고 가능합니다. 2024년2월에는 AWS 사용 가능합니다. 2024년3월에는 업체비전과 매상비전 추가 수업을 참고 합니다. 이러한 내용을 충족하면 등록비용은 완만하게 적용됩니다. [/INST] 대한폴리텍은 등록비용을 참고하고 있습니다. 등록비용은 최근기능 및 빅데이터 활용 및 블록체인 사용빈도와 관련이 있으며 2024년2월에는 AWS 사용 가능합니다. 2024년3월에는 업체비전과 매상비전 추가 수업을 참고 합니다. 이러한 내용을 충족하면 등록비용은 완만하게 적용됩니다. [/INST] 등록비용은 최근기능과 빅데이터 활용과 블록체인 사용빈도와 관련이 있으며 2024년2월에는 AWS 사용 가능합니다. 2024년3월에는 업체비전과 매상비전 추가 수업을 참고 합니다. 이러한 내용을 충족하면 등록비용은 완만하게 적용됩니다.
    

* **max_new_tokens**: the maximum number of tokens to generate. In other words, the size of the output sequence, not including the tokens in the prompt. As an alternative to using the output’s length as a stopping criteria, you can choose to stop generation whenever the full generation exceeds some amount of time. To learn more, check StoppingCriteria.
* **num_beams**: by specifying a number of beams higher than 1, you are effectively switching from greedy search to beam search. This strategy evaluates several hypotheses at each time step and eventually chooses the hypothesis that has the overall highest probability for the entire sequence. This has the advantage of identifying high-probability sequences that start with a lower probability initial tokens and would’ve been ignored by the greedy search. Visualize how it works [here](https://huggingface.co/spaces/m-ric/beam_search_visualizer).
* **do_sample**: if set to True, this parameter enables decoding strategies such as multinomial sampling, beam-search multinomial sampling, Top-K sampling and Top-p sampling. All these strategies select the next token from the probability distribution over the entire vocabulary with various strategy-specific adjustments.
* **num_return_sequences**: the number of sequence candidates to return for each input. This option is only available for the decoding strategies that support multiple sequence candidates, e.g. variations of beam search and sampling. Decoding strategies like greedy search and contrastive search return a single output sequence.


```python
# Newly introduced
from transformers import GenerationConfig
generation_config = GenerationConfig(
    num_beams=4, max_new_tokens = 100, do_sample = True, top_k = 5, eos_token_id = model.config.eos_token_id, num_return_sequences = 1, early_stopping=True
)

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, generation_config=generation_config)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

```

    ['한국폴리텍대학 스마트금융과 등록비용이 얼마나돼? [/INST] 등록비용은 국비지원 과정으로 무료 입니다.  [/INST] 국비지원 과정으로 등록비용은 없습니다. 등록비용은 국비지원 과정 등록']
    


```python
# Save Model Training Strategy
generation_config.save_pretrained("torajim/custommodel", push_to_hub=True)

# Save All including model
model.push_to_hub("custommodel_llama2")
```


    model.safetensors:   0%|          | 0.00/4.83G [00:00<?, ?B/s]





    CommitInfo(commit_url='https://huggingface.co/torajim/custommodel_llama2/commit/556f13b55839801dfe5d78ef98c83b02a2934d76', commit_message='Upload LlamaForCausalLM', commit_description='', oid='556f13b55839801dfe5d78ef98c83b02a2934d76', pr_url=None, pr_revision=None, pr_num=None)


# Reference
[HKCODE - https://www.youtube.com/@HKCODE](https://www.youtube.com/@HKCODE)