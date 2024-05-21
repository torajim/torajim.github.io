---
layout: single
title:  "[Assistants API - 3/4] GPT-4o에서 file search(vector store) 사용"
categories: openai
tag: [openai, gpt-4o, assistant, python, file-search]
toc: true
---

## Assistant + tools
- **code_interpreter**
   - 토큰 길이에 상관 없이, 코드를 생성해서 답변할 수 있는 도구 입니다.
   - 예: 피보나치 수열을 만들어줘 -> 피보나치 수열을 생성하는 함수를 생성한 후 이것의 실행 결과로 답변을 만듦 (cf., Code Interpreter가 아니라면, text 기반의 sequence generation으로 답변을 함)
   - 함수를 생성하는 방식이기 때문에, input, output의 token length 제약을 벗어나서 활용을 할 수 있습니다.
   - 데이터와 그래프 이미지가 포함된 파일을 생성할 수 있습니다.
- **[이번 내용] file_search**
   - 우리가 업로드한 자료에서 검색한 이후 답변을 만들어 줍니다. (RAG를 활용하는 것과 비슷)
   - Foundation model을 sFT하는데 걸리는 시간을 생각하면, 대부분의 짧은 배치는 RAG 형태로 구성하고, 긴배치(ex., 1~2주일에 한번)는 sFT를 하는 방식이 적절할 것으로 보입니다.
   - File을 Vector Store에 넣어두고 Assistant가 이를 활용할 수 있습니다.
   - 파일 하나당 최대 크기 512MB, 최대 토큰수 5,000,000까지 지원합니다.
   - [지원하는 파일 타입](https://platform.openai.com/docs/assistants/tools/file-search/supported-files)
   - [Known Limitation](https://platform.openai.com/docs/assistants/tools/file-search/how-it-works)
   - 아직 이미지 파일이나, csv, jsonl을 지원하지 않는데 추후 업데이트 될 것으로 보입니다. (24-05-21 기준)
- **function**
   - 우리가 직접 만든 함수를 사용할 수 있게 합니다.

## 기본 작업
- Open AI 라이브러리 설치
- Dotenv 설치
- Google Drive 연결
- API Key 로드
- Open AI 클라이언트 객체 생성


```python
# 실행을 위한 기본 설치 및 로드
!pip install --upgrade openai -q
!pip show openai | grep Version
!pip install -U python-dotenv
from dotenv import load_dotenv
import json
from openai import OpenAI
import os
from google.colab import drive
```

    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m320.6/320.6 kB[0m [31m2.3 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m75.6/75.6 kB[0m [31m3.9 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m77.9/77.9 kB[0m [31m1.7 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m58.3/58.3 kB[0m [31m5.0 MB/s[0m eta [36m0:00:00[0m
    [?25hVersion: 1.30.1
    Collecting python-dotenv
      Downloading python_dotenv-1.0.1-py3-none-any.whl (19 kB)
    Installing collected packages: python-dotenv
    Successfully installed python-dotenv-1.0.1
    


```python
drive.mount('/content/drive', force_remount=True)

load_dotenv(
    dotenv_path='drive/MyDrive/AUTH/.env',
    verbose=True,
)
api_key = os.environ.get("OPEN_API_KEY")
client = OpenAI(api_key=api_key)

# 객체 내용 출력 유틸
def show_json(obj):
  display(json.loads(obj.model_dump_json()))

def _get_response(thread_id):
    return client.beta.threads.messages.list(thread_id=thread_id, order="asc")

# Thread message 출력 유틸
def print_message(thread_id):
    for res in _get_response(thread_id):
        print(f"[{res.role.upper()}]\n{res.content[0].text.value}\n")
```

    Mounted at /content/drive
    

## File-Search Assistant 생성


```python
assistant = client.beta.assistants.create(
  name="Urban Planning Assistant",
  instructions="You are an expert of urban planning and designing. Use your knowledge base to answer questions about city roadmap related questions.",
  model="gpt-4o",
  tools=[{"type": "file_search"}]
)
show_json(assistant)
```


    {'id': 'asst_LKlEuVDN6Q3pg6HrwOeJOLSg',
     'created_at': 1716254498,
     'description': None,
     'instructions': 'You are an expert of urban planning and designing. Use your knowledge base to answer questions about city roadmap related questions.',
     'metadata': {},
     'model': 'gpt-4o',
     'name': 'Urban Planning Assistant',
     'object': 'assistant',
     'tools': [{'type': 'file_search'}],
     'response_format': 'auto',
     'temperature': 1.0,
     'tool_resources': {'code_interpreter': None,
      'file_search': {'vector_store_ids': []}},
     'top_p': 1.0}


## Upload files and add them to a Vector Store
- Vector store를 하나 만들고 파일을 업로드 한 후 모든 파일의 업로드 상태가 종료되었는지 확인해야 합니다.


```python
vector_store = client.beta.vector_stores.create(
    name="my_first_store"
)

# Ready the files for upload to OpenAI
# PDF 파일도 이미지로만 된 것은 upload시 fail하고, vector화 된 것만 들어감
file_paths = ["drive/MyDrive/data/(2030년)수원도시기본계획.pdf", "drive/MyDrive/data/수원시-스마트도시계획-본보고서.pdf"]
file_streams = [open(path, "rb") for path in file_paths]

# Use the upload and poll SDK helper to upload the files, add them to the vector store,
# and poll the status of the file batch for completion.
file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
  vector_store_id = vector_store.id, files=file_streams,
)

# You can print the status and the file counts of the batch to see the result of this operation.
print(file_batch.status)
print(file_batch.file_counts)

show_json(vector_store)
```

    completed
    FileCounts(cancelled=0, completed=1, failed=1, in_progress=0, total=2)
    


    {'id': 'vs_GU0BVMfMjeRwxMJByreSLLZR',
     'created_at': 1716258891,
     'file_counts': {'cancelled': 0,
      'completed': 0,
      'failed': 0,
      'in_progress': 0,
      'total': 0},
     'last_active_at': 1716258891,
     'metadata': {},
     'name': 'my_first_store',
     'object': 'vector_store',
     'status': 'completed',
     'usage_bytes': 0,
     'expires_after': None,
     'expires_at': None}


## Assistant Update
- 생성한 vector store를 참조할 수 있도록 업데이트 합니다.
- 만약 기존에 만들어 놓은 vector store가 있다면 assistant 생성 시점부터 이를 활용하도록 설정할 수 있습니다.
- [Storage - OpenAI API](https://platform.openai.com/storage/vector_stores) 에서 기존에 생성한 Vector Store의 id를 확인할 수 있습니다.


```python
assistant = client.beta.assistants.update(
  assistant_id=assistant.id,
  tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
)
show_json(assistant)
```


    {'id': 'asst_LKlEuVDN6Q3pg6HrwOeJOLSg',
     'created_at': 1716254498,
     'description': None,
     'instructions': 'You are an expert of urban planning and designing. Use your knowledge base to answer questions about city roadmap related questions.',
     'metadata': {},
     'model': 'gpt-4o',
     'name': 'Urban Planning Assistant',
     'object': 'assistant',
     'tools': [{'type': 'file_search'}],
     'response_format': 'auto',
     'temperature': 1.0,
     'tool_resources': {'code_interpreter': None,
      'file_search': {'vector_store_ids': ['vs_GU0BVMfMjeRwxMJByreSLLZR']}},
     'top_p': 1.0}


## Thread 생성 및 실행
- Thread, Message, Run 차례로 생성 및 실행


```python
thread = client.beta.threads.create()

message = client.beta.threads.messages.create(
    thread_id = thread.id,
    role = "user",
    content= "수원시의 미래 교통 환경은 어떻게 변화될지 설명해줘",
)

run = client.beta.threads.runs.create_and_poll(
    thread_id = thread.id,
    assistant_id = assistant.id,
)
show_json(run)
```


    {'id': 'run_akMb8u1qnqGYYtrEWuleqvel',
     'assistant_id': 'asst_LKlEuVDN6Q3pg6HrwOeJOLSg',
     'cancelled_at': None,
     'completed_at': 1716259372,
     'created_at': 1716259359,
     'expires_at': None,
     'failed_at': None,
     'incomplete_details': None,
     'instructions': 'You are an expert of urban planning and designing. Use your knowledge base to answer questions about city roadmap related questions.',
     'last_error': None,
     'max_completion_tokens': None,
     'max_prompt_tokens': None,
     'metadata': {},
     'model': 'gpt-4o',
     'object': 'thread.run',
     'required_action': None,
     'response_format': 'auto',
     'started_at': 1716259360,
     'status': 'completed',
     'thread_id': 'thread_Dimy3eqmr9KWeDr1F02hdmBR',
     'tool_choice': 'auto',
     'tools': [{'type': 'file_search'}],
     'truncation_strategy': {'type': 'auto', 'last_messages': None},
     'usage': {'completion_tokens': 608,
      'prompt_tokens': 12195,
      'total_tokens': 12803},
     'temperature': 1.0,
     'top_p': 1.0,
     'tool_resources': {}}



```python
if run.status == 'completed':
  messages = client.beta.threads.messages.list(
    thread_id=thread.id
  )
  show_json(messages)
  print_message(thread.id)
else:
  print(run.status)
```


    {'data': [{'id': 'msg_0szkugWU7WNmjFgRQLMYhUDO',
       'assistant_id': 'asst_LKlEuVDN6Q3pg6HrwOeJOLSg',
       'attachments': [],
       'completed_at': None,
       'content': [{'text': {'annotations': [{'end_index': 236,
            'file_citation': {'file_id': 'file-yPoNrzrNh3qE4Z8RBqcU8xgH',
             'quote': None},
            'start_index': 224,
            'text': '【4:0†source】',
            'type': 'file_citation'},
           {'end_index': 329,
            'file_citation': {'file_id': 'file-yPoNrzrNh3qE4Z8RBqcU8xgH',
             'quote': None},
            'start_index': 317,
            'text': '【4:3†source】',
            'type': 'file_citation'},
           {'end_index': 342,
            'file_citation': {'file_id': 'file-yPoNrzrNh3qE4Z8RBqcU8xgH',
             'quote': None},
            'start_index': 329,
            'text': '【4:11†source】',
            'type': 'file_citation'},
           {'end_index': 492,
            'file_citation': {'file_id': 'file-yPoNrzrNh3qE4Z8RBqcU8xgH',
             'quote': None},
            'start_index': 480,
            'text': '【4:1†source】',
            'type': 'file_citation'},
           {'end_index': 504,
            'file_citation': {'file_id': 'file-yPoNrzrNh3qE4Z8RBqcU8xgH',
             'quote': None},
            'start_index': 492,
            'text': '【4:5†source】',
            'type': 'file_citation'},
           {'end_index': 645,
            'file_citation': {'file_id': 'file-yPoNrzrNh3qE4Z8RBqcU8xgH',
             'quote': None},
            'start_index': 633,
            'text': '【4:1†source】',
            'type': 'file_citation'},
           {'end_index': 658,
            'file_citation': {'file_id': 'file-yPoNrzrNh3qE4Z8RBqcU8xgH',
             'quote': None},
            'start_index': 645,
            'text': '【4:12†source】',
            'type': 'file_citation'},
           {'end_index': 758,
            'file_citation': {'file_id': 'file-yPoNrzrNh3qE4Z8RBqcU8xgH',
             'quote': None},
            'start_index': 745,
            'text': '【4:13†source】',
            'type': 'file_citation'},
           {'end_index': 824,
            'file_citation': {'file_id': 'file-yPoNrzrNh3qE4Z8RBqcU8xgH',
             'quote': None},
            'start_index': 811,
            'text': '【4:11†source】',
            'type': 'file_citation'},
           {'end_index': 837,
            'file_citation': {'file_id': 'file-yPoNrzrNh3qE4Z8RBqcU8xgH',
             'quote': None},
            'start_index': 824,
            'text': '【4:14†source】',
            'type': 'file_citation'},
           {'end_index': 949,
            'file_citation': {'file_id': 'file-yPoNrzrNh3qE4Z8RBqcU8xgH',
             'quote': None},
            'start_index': 937,
            'text': '【4:6†source】',
            'type': 'file_citation'},
           {'end_index': 962,
            'file_citation': {'file_id': 'file-yPoNrzrNh3qE4Z8RBqcU8xgH',
             'quote': None},
            'start_index': 949,
            'text': '【4:15†source】',
            'type': 'file_citation'}],
          'value': '수원시의 미래 교통 환경은 다양한 친환경 및 첨단 교통수단 도입과 교통체계 개선을 통해 크게 변화할 것으로 예상됩니다. 다음은 주요 변화를 설명한 내용입니다.\n\n1. **친환경 교통수단 도입 및 확대**:\n   - 수원시는 비동력 및 무탄소 교통수단의 수송 분담률을 증가시키기 위해 노력할 것입니다. 이를 통해 환경 친화적인 에너지 절감형 교통물류체계로 전환하고 기후변화에 적극적으로 대처할 계획입니다【4:0†source】.\n   - 미래 교통수단으로 노면전차, 전기버스, 바이모달 트램 등의 도입을 검토하고, 자전거 도로를 확장하여 걷기 좋은 거리를 넓혀갈 것입니다【4:3†source】【4:11†source】.\n\n2. **교통 인프라 개선**:\n   - 광역교통망 계획에서는 고속도로 및 주요 간선도로의 광역도로망 연계 기능을 강화할 예정입니다. 예를 들어, 평택화성 고속도로, 봉담∼동탄 고속도로 등이 수원시와 경기 남부지역 간의 연계를 강화할 것입니다【4:1†source】【4:5†source】.\n\n3. **지능형 교통 시스템(ITS) 도입**:\n   - 지능형 교통 시스템(ITS) 구축을 통해 시설 간 연계를 효율화하고, 보행자전용도로와 자전거도로 등 에너지 절약형 교통망을 확충하여 친환경적 교통체계를 수립할 계획입니다【4:1†source】【4:12†source】.\n\n4. **자전거와 보행 환경 개선**:\n   - 자전거 도로망 확충 및 자전거 보관대 설치, 자전거 이용시설 확충을 통해 자전거 이용을 촉진할 것입니다【4:13†source】.\n   - 보행 환경 개선을 통해 지속 가능한 생태 교통체계를 구축하는 것도 중요한 목표입니다【4:11†source】【4:14†source】.\n\n5. **효율적인 환승 체계 구축**:\n   - 주요 교통 환승센터 예를 들어 수원역에 환승센터를 조성하여 빠르고 편리한 도로 환경 및 대중교통 연계 체계를 구축할 예정입니다【4:6†source】【4:15†source】.\n\n이러한 계획들은 수원시가 친환경적이고 지속 가능한 교통체계를 구축하여 시민의 편의와 환경 보호를 동시에 달성할 수 있도록 돕는 중요한 변화들입니다.'},
         'type': 'text'}],
       'created_at': 1716259362,
       'incomplete_at': None,
       'incomplete_details': None,
       'metadata': {},
       'object': 'thread.message',
       'role': 'assistant',
       'run_id': 'run_akMb8u1qnqGYYtrEWuleqvel',
       'status': None,
       'thread_id': 'thread_Dimy3eqmr9KWeDr1F02hdmBR'},
      {'id': 'msg_AR4HHbuFpkLsU0mpkntpZ1Tj',
       'assistant_id': None,
       'attachments': [],
       'completed_at': None,
       'content': [{'text': {'annotations': [],
          'value': '수원시의 미래 교통 환경은 어떻게 변화될지 설명해줘'},
         'type': 'text'}],
       'created_at': 1716259359,
       'incomplete_at': None,
       'incomplete_details': None,
       'metadata': {},
       'object': 'thread.message',
       'role': 'user',
       'run_id': None,
       'status': None,
       'thread_id': 'thread_Dimy3eqmr9KWeDr1F02hdmBR'}],
     'object': 'list',
     'first_id': 'msg_0szkugWU7WNmjFgRQLMYhUDO',
     'last_id': 'msg_AR4HHbuFpkLsU0mpkntpZ1Tj',
     'has_more': False}


    [USER]
    수원시의 미래 교통 환경은 어떻게 변화될지 설명해줘
    
    [ASSISTANT]
    수원시의 미래 교통 환경은 다양한 친환경 및 첨단 교통수단 도입과 교통체계 개선을 통해 크게 변화할 것으로 예상됩니다. 다음은 주요 변화를 설명한 내용입니다.
    
    1. **친환경 교통수단 도입 및 확대**:
       - 수원시는 비동력 및 무탄소 교통수단의 수송 분담률을 증가시키기 위해 노력할 것입니다. 이를 통해 환경 친화적인 에너지 절감형 교통물류체계로 전환하고 기후변화에 적극적으로 대처할 계획입니다【4:0†source】.
       - 미래 교통수단으로 노면전차, 전기버스, 바이모달 트램 등의 도입을 검토하고, 자전거 도로를 확장하여 걷기 좋은 거리를 넓혀갈 것입니다【4:3†source】【4:11†source】.
    
    2. **교통 인프라 개선**:
       - 광역교통망 계획에서는 고속도로 및 주요 간선도로의 광역도로망 연계 기능을 강화할 예정입니다. 예를 들어, 평택화성 고속도로, 봉담∼동탄 고속도로 등이 수원시와 경기 남부지역 간의 연계를 강화할 것입니다【4:1†source】【4:5†source】.
    
    3. **지능형 교통 시스템(ITS) 도입**:
       - 지능형 교통 시스템(ITS) 구축을 통해 시설 간 연계를 효율화하고, 보행자전용도로와 자전거도로 등 에너지 절약형 교통망을 확충하여 친환경적 교통체계를 수립할 계획입니다【4:1†source】【4:12†source】.
    
    4. **자전거와 보행 환경 개선**:
       - 자전거 도로망 확충 및 자전거 보관대 설치, 자전거 이용시설 확충을 통해 자전거 이용을 촉진할 것입니다【4:13†source】.
       - 보행 환경 개선을 통해 지속 가능한 생태 교통체계를 구축하는 것도 중요한 목표입니다【4:11†source】【4:14†source】.
    
    5. **효율적인 환승 체계 구축**:
       - 주요 교통 환승센터 예를 들어 수원역에 환승센터를 조성하여 빠르고 편리한 도로 환경 및 대중교통 연계 체계를 구축할 예정입니다【4:6†source】【4:15†source】.
    
    이러한 계획들은 수원시가 친환경적이고 지속 가능한 교통체계를 구축하여 시민의 편의와 환경 보호를 동시에 달성할 수 있도록 돕는 중요한 변화들입니다.
    
    

## 실행 단계를 살펴보기 위해 Steps 활용
- 질문에 대해 답을 만들 때 file search를 사용한 것을 확인할 수 있습니다.


```python
run_steps = client.beta.threads.runs.steps.list(
    thread_id = thread.id, run_id = run.id, order="asc",
)
show_json(run_steps)
```


    {'data': [{'id': 'step_hDfCs6pWzLuacZ3sOvgt27vo',
       'assistant_id': 'asst_LKlEuVDN6Q3pg6HrwOeJOLSg',
       'cancelled_at': None,
       'completed_at': 1716259362,
       'created_at': 1716259361,
       'expired_at': None,
       'failed_at': None,
       'last_error': None,
       'metadata': None,
       'object': 'thread.run.step',
       'run_id': 'run_akMb8u1qnqGYYtrEWuleqvel',
       'status': 'completed',
       'step_details': {'tool_calls': [{'id': 'call_NIMlM37cvPIDjb3990qKlPKk',
          'file_search': {},
          'type': 'file_search'}],
        'type': 'tool_calls'},
       'thread_id': 'thread_Dimy3eqmr9KWeDr1F02hdmBR',
       'type': 'tool_calls',
       'usage': {'completion_tokens': 18,
        'prompt_tokens': 639,
        'total_tokens': 657},
       'expires_at': None},
      {'id': 'step_NpJNEMRNGR75eBNCKPzjw6Zd',
       'assistant_id': 'asst_LKlEuVDN6Q3pg6HrwOeJOLSg',
       'cancelled_at': None,
       'completed_at': 1716259372,
       'created_at': 1716259362,
       'expired_at': None,
       'failed_at': None,
       'last_error': None,
       'metadata': None,
       'object': 'thread.run.step',
       'run_id': 'run_akMb8u1qnqGYYtrEWuleqvel',
       'status': 'completed',
       'step_details': {'message_creation': {'message_id': 'msg_0szkugWU7WNmjFgRQLMYhUDO'},
        'type': 'message_creation'},
       'thread_id': 'thread_Dimy3eqmr9KWeDr1F02hdmBR',
       'type': 'message_creation',
       'usage': {'completion_tokens': 590,
        'prompt_tokens': 11556,
        'total_tokens': 12146},
       'expires_at': None}],
     'object': 'list',
     'first_id': 'step_hDfCs6pWzLuacZ3sOvgt27vo',
     'last_id': 'step_NpJNEMRNGR75eBNCKPzjw6Zd',
     'has_more': False}


## Reference
- [File Search - Assistants API(Beta) - Open AI](https://platform.openai.com/docs/assistants/tools/file-search)
- [EP02. #openai의 새로운 기능 #assistant API 3가지 도구 활용법 - 테디노트](https://www.youtube.com/watch?v=BMW1NJkL7Ks)
- [소스 코드 참조 - Colab](https://colab.research.google.com/drive/1bGtWYX5TQ3uqzEMFdzTGW9JUD-iMB6O2?usp=sharing)
