---
layout: single
title:  "[Assistants API - 2/4] Open AI Assistants API에서 code interpreter 사용"
categories: openai
tag: [openai, gpt-4o, assistant, python, code-interpreter]
toc: true
---

## Assistant + tools
- **[이번 내용] code_interpreter**
   - 토큰 길이에 상관 없이, 코드를 생성해서 답변할 수 있는 도구 입니다.
   - 예: 피보나치 수열을 만들어줘 -> 피보나치 수열을 생성하는 함수를 생성한 후 이것의 실행 결과로 답변을 만듦 (cf., Code Interpreter가 아니라면, text 기반의 sequence generation으로 답변을 함)
   - 함수를 생성하는 방식이기 때문에, input, output의 token length 제약을 벗어나서 활용을 할 수 있습니다.
   - 데이터와 그래프 이미지가 포함된 파일을 생성할 수 있습니다.
- **file_search**
   - 우리가 업로드한 자료에서 검색한 이후 답변을 만들어 줍니다. (RAG를 활용하는 것과 비슷)
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

    Version: 1.30.1
    Requirement already satisfied: python-dotenv in /usr/local/lib/python3.10/dist-packages (1.0.1)
    


```python
drive.mount('/content/drive', force_remount=True)

load_dotenv(
    dotenv_path='drive/MyDrive/AUTH/.env',
    verbose=True,
)
api_key = os.environ.get("OPEN_API_KEY")
client = OpenAI(api_key=api_key)

def show_json(obj):
  display(json.loads(obj.model_dump_json()))
```

    Mounted at /content/drive
    

## Code Interpreter Assistant 생성


```python
assistant = client.beta.assistants.create(
  instructions="You are a personal math tutor. When asked a math question, write and run code to answer the question.",
  model="gpt-4o",
  tools=[{"type": "code_interpreter"}]
)
show_json(assistant)
```


    {'id': 'asst_Y7Ndt1RKCg2Cb4mhlpgVbGbS',
     'created_at': 1716196766,
     'description': None,
     'instructions': 'You are a personal math tutor. When asked a math question, write and run code to answer the question.',
     'metadata': {},
     'model': 'gpt-4o',
     'name': None,
     'object': 'assistant',
     'tools': [{'type': 'code_interpreter'}],
     'response_format': 'auto',
     'temperature': 1.0,
     'tool_resources': {'code_interpreter': {'file_ids': []}, 'file_search': None},
     'top_p': 1.0}


## Thread 생성 및 실행
- Thread, Message, Run 차례로 생성 및 실행


```python
thread = client.beta.threads.create()

message = client.beta.threads.messages.create(
    thread_id = thread.id,
    role = "user",
    content= "코드를 사용해서 최초 20개의 피보나치 수열을 만들어줘",
)

run = client.beta.threads.runs.create_and_poll(
    thread_id = thread.id,
    assistant_id = assistant.id,
)
show_json(run)
```


    {'id': 'run_pqOAuhLfHIBmPXJ7ttkNF6Gw',
     'assistant_id': 'asst_Y7Ndt1RKCg2Cb4mhlpgVbGbS',
     'cancelled_at': None,
     'completed_at': 1716197343,
     'created_at': 1716197337,
     'expires_at': None,
     'failed_at': None,
     'incomplete_details': None,
     'instructions': 'You are a personal math tutor. When asked a math question, write and run code to answer the question.',
     'last_error': None,
     'max_completion_tokens': None,
     'max_prompt_tokens': None,
     'metadata': {},
     'model': 'gpt-4o',
     'object': 'thread.run',
     'required_action': None,
     'response_format': 'auto',
     'started_at': 1716197337,
     'status': 'completed',
     'thread_id': 'thread_AMQPcatbKuhqL8SMsTggLkdl',
     'tool_choice': 'auto',
     'tools': [{'type': 'code_interpreter'}],
     'truncation_strategy': {'type': 'auto', 'last_messages': None},
     'usage': {'completion_tokens': 239,
      'prompt_tokens': 486,
      'total_tokens': 725},
     'temperature': 1.0,
     'top_p': 1.0,
     'tool_resources': {}}



```python
if run.status == 'completed':
  messages = client.beta.threads.messages.list(
    thread_id=thread.id
  )
  show_json(messages)
else:
  print(run.status)
```


    {'data': [{'id': 'msg_ETt57jde3ZqDb0Bs0K2XEiWN',
       'assistant_id': 'asst_Y7Ndt1RKCg2Cb4mhlpgVbGbS',
       'attachments': [],
       'completed_at': None,
       'content': [{'text': {'annotations': [],
          'value': '최초 20개의 피보나치 수열은 다음과 같습니다:\n\n\\[ 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181 \\]\n\n이 수열은 각 숫자가 바로 앞의 두 숫자의 합으로 구성됩니다. 예를 들어 2는 1 + 1, 3은 1 + 2, 5는 2 + 3입니다.'},
         'type': 'text'}],
       'created_at': 1716197341,
       'incomplete_at': None,
       'incomplete_details': None,
       'metadata': {},
       'object': 'thread.message',
       'role': 'assistant',
       'run_id': 'run_pqOAuhLfHIBmPXJ7ttkNF6Gw',
       'status': None,
       'thread_id': 'thread_AMQPcatbKuhqL8SMsTggLkdl'},
      {'id': 'msg_QLaWfYg2AYhFGuK6pcqDfl3Q',
       'assistant_id': None,
       'attachments': [],
       'completed_at': None,
       'content': [{'text': {'annotations': [],
          'value': '코드를 사용해서 최초 20개의 피보나치 수열을 만들어줘'},
         'type': 'text'}],
       'created_at': 1716197337,
       'incomplete_at': None,
       'incomplete_details': None,
       'metadata': {},
       'object': 'thread.message',
       'role': 'user',
       'run_id': None,
       'status': None,
       'thread_id': 'thread_AMQPcatbKuhqL8SMsTggLkdl'}],
     'object': 'list',
     'first_id': 'msg_ETt57jde3ZqDb0Bs0K2XEiWN',
     'last_id': 'msg_QLaWfYg2AYhFGuK6pcqDfl3Q',
     'has_more': False}


## 실행 단계를 살펴보기 위해 Steps 활용
- 질문에 대해 답을 만들 때 실제 함수를 생성하는 것을 확인 할 수 있음


```python
run_steps = client.beta.threads.runs.steps.list(
    thread_id = thread.id, run_id = run.id, order="asc",
)
show_json(run_steps)
```


    {'data': [{'id': 'step_9I2fBWyMtivKvaLauynEph6j',
       'assistant_id': 'asst_Y7Ndt1RKCg2Cb4mhlpgVbGbS',
       'cancelled_at': None,
       'completed_at': 1716197341,
       'created_at': 1716197338,
       'expired_at': None,
       'failed_at': None,
       'last_error': None,
       'metadata': None,
       'object': 'thread.run.step',
       'run_id': 'run_pqOAuhLfHIBmPXJ7ttkNF6Gw',
       'status': 'completed',
       'step_details': {'tool_calls': [{'id': 'call_rVAbMWt9OhKznoszzP9VBYVS',
          'code_interpreter': {'input': '# 피보나치 수열의 최초 20개 숫자를 생성하는 함수\ndef generate_fibonacci(n):\n    fibonacci_sequence = [0, 1]\n    while len(fibonacci_sequence) < n:\n        next_value = fibonacci_sequence[-1] + fibonacci_sequence[-2]\n        fibonacci_sequence.append(next_value)\n    return fibonacci_sequence\n\n# 최초 20개의 피보나치 수열 생성\nfirst_20_fibonacci = generate_fibonacci(20)\nfirst_20_fibonacci',
           'outputs': [{'logs': '[0,\n 1,\n 1,\n 2,\n 3,\n 5,\n 8,\n 13,\n 21,\n 34,\n 55,\n 89,\n 144,\n 233,\n 377,\n 610,\n 987,\n 1597,\n 2584,\n 4181]',
             'type': 'logs'}]},
          'type': 'code_interpreter'}],
        'type': 'tool_calls'},
       'thread_id': 'thread_AMQPcatbKuhqL8SMsTggLkdl',
       'type': 'tool_calls',
       'usage': {'completion_tokens': 107,
        'prompt_tokens': 154,
        'total_tokens': 261},
       'expires_at': None},
      {'id': 'step_vWH14ewi199t8aTMBx7GLEoV',
       'assistant_id': 'asst_Y7Ndt1RKCg2Cb4mhlpgVbGbS',
       'cancelled_at': None,
       'completed_at': 1716197343,
       'created_at': 1716197341,
       'expired_at': None,
       'failed_at': None,
       'last_error': None,
       'metadata': None,
       'object': 'thread.run.step',
       'run_id': 'run_pqOAuhLfHIBmPXJ7ttkNF6Gw',
       'status': 'completed',
       'step_details': {'message_creation': {'message_id': 'msg_ETt57jde3ZqDb0Bs0K2XEiWN'},
        'type': 'message_creation'},
       'thread_id': 'thread_AMQPcatbKuhqL8SMsTggLkdl',
       'type': 'message_creation',
       'usage': {'completion_tokens': 132,
        'prompt_tokens': 332,
        'total_tokens': 464},
       'expires_at': None}],
     'object': 'list',
     'first_id': 'step_9I2fBWyMtivKvaLauynEph6j',
     'last_id': 'step_vWH14ewi199t8aTMBx7GLEoV',
     'has_more': False}


## Steps 내용 중 input에 있는 code_interpreter가 생성한 코드 copy and paste
- print 구문과 함께 써 보면 개행 문자 포함 출력됨


```python
print(run_steps.data[0].step_details.tool_calls[0].code_interpreter.input)
```

    # 피보나치 수열의 최초 20개 숫자를 생성하는 함수
    def generate_fibonacci(n):
        fibonacci_sequence = [0, 1]
        while len(fibonacci_sequence) < n:
            next_value = fibonacci_sequence[-1] + fibonacci_sequence[-2]
            fibonacci_sequence.append(next_value)
        return fibonacci_sequence
    
    # 최초 20개의 피보나치 수열 생성
    first_20_fibonacci = generate_fibonacci(20)
    first_20_fibonacci
    


```python
def generate_fibonacci(n):
    fibonacci_sequence = [0, 1]
    while len(fibonacci_sequence) < n:
        next_value = fibonacci_sequence[-1] + fibonacci_sequence[-2]
        fibonacci_sequence.append(next_value)
    return fibonacci_sequence

# 최초 20개의 피보나치 수열 생성
first_20_fibonacci = generate_fibonacci(20)
first_20_fibonacci
```




    [0,
     1,
     1,
     2,
     3,
     5,
     8,
     13,
     21,
     34,
     55,
     89,
     144,
     233,
     377,
     610,
     987,
     1597,
     2584,
     4181]



## Reference
- [Assistants API(Beta) - Open AI](https://platform.openai.com/docs/assistants/overview)
- [EP02. #openai의 새로운 기능 #assistant API 3가지 도구 활용법 - 테디노트](https://www.youtube.com/watch?v=BMW1NJkL7Ks)
- [소스 코드 확인 - Colab](https://colab.research.google.com/drive/1xz4e7eBVNS5cO83-lOe-gh_iUp5ZVAOx?usp=sharing)