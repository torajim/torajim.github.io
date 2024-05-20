---
layout: single
title:  "[Assistants API - 1/4] Open AI Assistants API로 GPT-4o 모델에 질문 답변 받기"
categories: openai
tag: [openai, gpt-4o, assistant, python]
toc: true
---

## Assistant 작성
- [Open AI API - Assistants API](https://platform.openai.com/docs/assistants/overview)
- 생성 절차
1. 어시스턴트 정의: 사용자 지침을 설정하고 모델을 선택하여 어시스턴트를 정의합니다. 필요한 경우 파일을 추가하고 코드 인터프리터, 파일 검색, 함수 호출 같은 도구를 활성화할 수 있습니다.
2. 스레드 생성: 사용자가 대화를 시작하면 스레드를 생성합니다. 스레드는 대화의 흐름을 유지하고 관리하는 역할을 합니다. (i.e., 카카오의 채팅방)
3. 메시지 추가: 사용자가 질문할 때마다 해당 질문을 스레드에 메시지로 추가합니다. (메시지는 시스템 메시지, 사용자 메시지, 어시스턴트 메시지로 구분됨)
4. 어시스턴트 실행: 스레드에서 어시스턴트를 실행하여 모델과 도구를 호출하여 사용자 질문에 대한 응답을 생성합니다. (Assistant ID, Thread ID를 지정하는 방식)


![image](/assets/images/diagram-assistant.webp)

## Open AI 라이브러리 설치


```python
!pip install --upgrade openai -q
!pip show openai | grep Version
```

    Version: 1.30.1
    

## Helper 함수


```python
import json
def show_json(obj):
  display(json.loads(obj.model_dump_json()))
```

## Dotenv 설치
- .env 파일 내에 secret string 위치 시키고 코드 내에서 가져다 쓸 용도 입니다.


```python
!pip install -U python-dotenv
```

    Requirement already satisfied: python-dotenv in /usr/local/lib/python3.10/dist-packages (1.0.1)
    

## API Key 설정


```python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.environ.get("OPEN_API_KEY")
```

## Open AI 클라이언트 객체 생성


```python
from openai import OpenAI
client = OpenAI(api_key=api_key)
```

## Assistant 객체 생성
- 어시스턴트는 model, instructions, tool 등 여러 매개변수를 사용하여 사용자의 메시지에 응답할 수 있도록 구성할 수 있는 엔터티를 나타냅니다.
- 생성한 assistant는 아래에서 확인 가능합니다.
- [playground](https://platform.openai.com/assistants)
- Open AI billing 상태가 paid user가 아닌 경우 gpt-3.5-turbo 까지 사용 가능하고 rate limit이 3 RPM 수준
- gpt-4 이상 사용 위해 open ai 계정에 비용 충전 필요
- [가격 페이지](https://openai.com/api/pricing/)


```python
assistant = client.beta.assistants.create(
  name="Math Tutor",
  instructions="You are a personal math tutor. Answer questions briefly, in a sentence or less.",
  model="gpt-4o",
)

show_json(assistant)
```


    {'id': 'asst_KDw7D2Uen9C2vfqdLnN9TfSQ',
     'created_at': 1716130677,
     'description': None,
     'instructions': 'You are a personal math tutor. Answer questions briefly, in a sentence or less.',
     'metadata': {},
     'model': 'gpt-4o',
     'name': 'Math Tutor',
     'object': 'assistant',
     'tools': [],
     'response_format': 'auto',
     'temperature': 1.0,
     'tool_resources': {'code_interpreter': None, 'file_search': None},
     'top_p': 1.0}



```python
ASSISTANT_ID = assistant.id
print(ASSISTANT_ID)
```

    asst_KDw7D2Uen9C2vfqdLnN9TfSQ
    

## Thread 생성 및 Message 추가


```python
thread = client.beta.threads.create()
show_json(thread)
```


    {'id': 'thread_CUann2OKjBZozpZCbulBShvL',
     'created_at': 1716130677,
     'metadata': {},
     'object': 'thread',
     'tool_resources': {'code_interpreter': None, 'file_search': None}}



```python
message = client.beta.threads.messages.create(
    thread_id = thread.id,
    role = "user",
    content= "다음의 방정식을 풀고 싶습니다. '3x + 11 = 14', 수학선생님 도와주실 수 있나요?",
)
show_json(message)
```


    {'id': 'msg_rTEPP3wsgDExRdRP3yCUXnVf',
     'assistant_id': None,
     'attachments': [],
     'completed_at': None,
     'content': [{'text': {'annotations': [],
        'value': "다음의 방정식을 풀고 싶습니다. '3x + 11 = 14', 수학선생님 도와주실 수 있나요?"},
       'type': 'text'}],
     'created_at': 1716130678,
     'incomplete_at': None,
     'incomplete_details': None,
     'metadata': {},
     'object': 'thread.message',
     'role': 'user',
     'run_id': None,
     'status': None,
     'thread_id': 'thread_CUann2OKjBZozpZCbulBShvL'}


## Run 만들기
- Streaming 방식에 따라 받아오는게 다름
- 여기선 Non-streaming 방식으로 구현


```python
run = client.beta.threads.runs.create_and_poll(
    thread_id = thread.id,
    assistant_id = ASSISTANT_ID,
)
show_json(run)
```


    {'id': 'run_73XLuBlR2WWpKfe4iw6GyzFZ',
     'assistant_id': 'asst_KDw7D2Uen9C2vfqdLnN9TfSQ',
     'cancelled_at': None,
     'completed_at': 1716130680,
     'created_at': 1716130678,
     'expires_at': None,
     'failed_at': None,
     'incomplete_details': None,
     'instructions': 'You are a personal math tutor. Answer questions briefly, in a sentence or less.',
     'last_error': None,
     'max_completion_tokens': None,
     'max_prompt_tokens': None,
     'metadata': {},
     'model': 'gpt-4o',
     'object': 'thread.run',
     'required_action': None,
     'response_format': 'auto',
     'started_at': 1716130679,
     'status': 'completed',
     'thread_id': 'thread_CUann2OKjBZozpZCbulBShvL',
     'tool_choice': 'auto',
     'tools': [],
     'truncation_strategy': {'type': 'auto', 'last_messages': None},
     'usage': {'completion_tokens': 61, 'prompt_tokens': 75, 'total_tokens': 136},
     'temperature': 1.0,
     'top_p': 1.0,
     'tool_resources': {}}


## Run의 상태 변화 정리

| 상태 | 정의 |
|-----|------|
| 대기 중 (queued) | 실행이 처음 생성되었거나 필요한 작업을 완료한 후 대기 중 상태로 이동합니다. <br>거의 즉시 진행 중 상태로 이동해야 합니다. |
| 진행 중 (in_progress) | 진행 중 상태에서는 어시스턴트가 모델과 도구를 사용하여 단계를 수행합니다. <br> 실행 단계를 검사하여 진행 상황을 볼 수 있습니다.              |
| 완료됨 (completed) | 실행이 성공적으로 완료되었습니다! <br>이제 어시스턴트가 스레드에 추가한 모든 메시지와 실행이 수행한 모든 단계를 볼 수 있습니다. <br>사용자 메시지를 추가하고 새 실행을 생성하여 대화를 계속할 수 있습니다. |
| 작업 필요 (requires_action) | 함수 호출 도구를 사용할 때, 모델이 호출할 함수의 이름과 인수를 결정하면 실행은 작업 필요 상태로 이동합니다. <br>그런 다음 해당 함수를 실행하고 출력을 제출해야 실행이 계속됩니다. <br>만약 expires_at 타임스탬프(생성 후 약 10분)가 지나기 전에 출력을 제공하지 않으면 실행은 만료 상태로 이동합니다. |
| 만료됨 (expired) | 함수 호출 출력이 expires_at 전에 제출되지 않으면 실행이 만료됩니다. <br>또한 실행이 너무 오래 걸려 expires_at 시간보다 초과되면 시스템이 실행을 만료시킵니다.  |
| 취소 중 (cancelling) | 취소 실행 엔드포인트를 사용하여 진행 중인 실행을 취소하려고 시도할 수 있습니다. <br>취소 시도가 성공하면 실행 상태는 취소됨으로 이동합니다. 취소가 시도되지만 보장되지는 않습니다. |
| 취소됨 (cancelled) | 실행이 성공적으로 취소되었습니다. |
| 실패함 (failed) | 실행 실패의 이유는 실행의 마지막 오류 객체를 확인하여 볼 수 있습니다. <br>실패 타임스탬프는 failed_at에 기록됩니다. |
| 불완전함 (incomplete) | 실행이 최대 프롬프트 토큰 또는 최대 완료 토큰에 도달하여 종료되었습니다. <br>불완전한 세부 사항 객체를 확인하여 구체적인 이유를 볼 수 있습니다. |



```python
if run.status == 'completed':
  messages = client.beta.threads.messages.list(
    thread_id=thread.id
  )
  show_json(messages)
else:
  print(run.status)
```


    {'data': [{'id': 'msg_Ulc6bpxmGcp8tzGeA8coHCYb',
       'assistant_id': 'asst_KDw7D2Uen9C2vfqdLnN9TfSQ',
       'attachments': [],
       'completed_at': None,
       'content': [{'text': {'annotations': [],
          'value': "네, 물론이죠! '3x + 11 = 14'를 풀려면,\n1. 양 변에서 11을 빼세요: '3x = 3'.\n2. 그리고 양 변을 3으로 나누세요: 'x = 1'."},
         'type': 'text'}],
       'created_at': 1716130679,
       'incomplete_at': None,
       'incomplete_details': None,
       'metadata': {},
       'object': 'thread.message',
       'role': 'assistant',
       'run_id': 'run_73XLuBlR2WWpKfe4iw6GyzFZ',
       'status': None,
       'thread_id': 'thread_CUann2OKjBZozpZCbulBShvL'},
      {'id': 'msg_rTEPP3wsgDExRdRP3yCUXnVf',
       'assistant_id': None,
       'attachments': [],
       'completed_at': None,
       'content': [{'text': {'annotations': [],
          'value': "다음의 방정식을 풀고 싶습니다. '3x + 11 = 14', 수학선생님 도와주실 수 있나요?"},
         'type': 'text'}],
       'created_at': 1716130678,
       'incomplete_at': None,
       'incomplete_details': None,
       'metadata': {},
       'object': 'thread.message',
       'role': 'user',
       'run_id': None,
       'status': None,
       'thread_id': 'thread_CUann2OKjBZozpZCbulBShvL'}],
     'object': 'list',
     'first_id': 'msg_Ulc6bpxmGcp8tzGeA8coHCYb',
     'last_id': 'msg_rTEPP3wsgDExRdRP3yCUXnVf',
     'has_more': False}


## 기존 메시지에 추가 질문


```python
message = client.beta.threads.messages.create(
    thread_id = thread.id,
    role = "user",
    content = "설명 감사합니다. 다른 비슷한 문제를 출제해 주시고 다시 설명해 주실 수 있나요?",
)

run = client.beta.threads.runs.create_and_poll(
    thread_id = thread.id,
    assistant_id = ASSISTANT_ID,
)

if run.status == 'completed':
  messages = client.beta.threads.messages.list(
    thread_id=thread.id,
    order="asc",
    after=message.id
    # after를 주면, 위 message 뒷부분만 가져옴(i.e., 답변만 가져옴)
  )
  show_json(messages)
else:
  print(run.status)
```


    {'data': [{'id': 'msg_V3rfpneFOlTwAHp0PEYjBGkH',
       'assistant_id': 'asst_KDw7D2Uen9C2vfqdLnN9TfSQ',
       'attachments': [],
       'completed_at': None,
       'content': [{'text': {'annotations': [],
          'value': "당연하죠! 다음 방정식을 풀어보세요: '6x - 10 = 20'.\n\n풀이 과정:\n1. 양 변에 10을 더하세요: '6x = 30'.\n2. 그리고 양 변을 6으로 나누세요: 'x = 5'."},
         'type': 'text'}],
       'created_at': 1716130797,
       'incomplete_at': None,
       'incomplete_details': None,
       'metadata': {},
       'object': 'thread.message',
       'role': 'assistant',
       'run_id': 'run_LNecBapvDLHz0EzTeqRv8v2u',
       'status': None,
       'thread_id': 'thread_CUann2OKjBZozpZCbulBShvL'}],
     'object': 'list',
     'first_id': 'msg_V3rfpneFOlTwAHp0PEYjBGkH',
     'last_id': 'msg_V3rfpneFOlTwAHp0PEYjBGkH',
     'has_more': False}


## Reference
- [Assistants API(Beta) - Open AI](https://platform.openai.com/docs/assistants/overview)
- [TeddyNote - EP01. #openai의 새로운 기능 #assistant API 완벽히 이해해보기](https://www.youtube.com/watch?v=-Wne4a-8RlY)
- [Colab Link](https://colab.research.google.com/drive/1rVx_2iESXD2FJf_4aPEpFJbUuDbtsp6z?usp=sharing)
