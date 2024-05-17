---
layout: single
title:  "Windows 패키지 관리자 Scoop 설치 및 사용법 안내"
categories: install
tag: [scoop, windows, powershell]
toc: true
---

## 소개

Scoop은 Windows 환경에서 소프트웨어 패키지를 쉽게 설치하고 관리할 수 있게 해주는 명령줄 기반 패키지 관리자입니다. Git, Python, Node.js 등 다양한 개발 도구를 간편하게 설치할 수 있습니다. (개인적인 경험상 일부 관리자 권한이 필요한 프로그램의 설치는 매끄럽지 않을 수 있습니다. 다만, Python, Java처럼 버저닝이 필요한 SDK를 설치할 때는 아주 깔끔한 느낌입니다. )
이 글에서는 Scoop의 설치 방법과 기본적인 사용법에 대해 알아보겠습니다.

## Scoop 설치 방법

### 1. PowerShell 실행

Scoop을 설치하기 위해서는 PowerShell을 관리자 권한으로 실행해야 합니다. Windows 검색창에 "PowerShell"을 입력하고, 검색 결과에서 "Windows PowerShell"을 마우스 오른쪽 버튼으로 클릭한 후 "관리자 권한으로 실행"을 선택합니다.

### 2. 실행 정책 설정

Scoop 설치 스크립트를 실행하기 전에 PowerShell의 실행 정책을 변경해야 합니다. 다음 명령어를 입력하여 실행 정책을 변경합니다:

```powershell
Set-ExecutionPolicy RemoteSigned -scope CurrentUser
```

이 명령어는 현재 사용자에 대해 원격으로 서명된 스크립트의 실행을 허용합니다.

### 3. Scoop 설치

실행 정책을 변경한 후, Scoop 설치 스크립트를 실행합니다. 다음 명령어를 입력합니다:

```powershell
iwr -useb get.scoop.sh | iex
```

이 명령어는 Scoop 설치 스크립트를 다운로드하고 실행합니다.

### 4. 설치 확인

Scoop이 정상적으로 설치되었는지 확인하려면 다음 명령어를 입력합니다:

```powershell
scoop help
```

이 명령어를 입력하면 Scoop의 도움말이 출력됩니다. 설치가 정상적으로 완료되었다면 다양한 Scoop 명령어와 사용법이 표시됩니다.

## Scoop 사용법

### 1. 패키지 설치

Scoop을 사용하여 패키지를 설치하는 것은 매우 간단합니다. 예를 들어, Git을 설치하려면 다음 명령어를 입력합니다:

```powershell
scoop install git
```

Scoop은 자동으로 모든 필요한 의존성을 관리하고, Git을 설치합니다.

### 2. 설치된 패키지 목록 확인

현재 설치된 패키지 목록을 확인하려면 다음 명령어를 입력합니다:

```powershell
scoop list
```

이 명령어는 Scoop을 통해 설치된 모든 패키지를 나열합니다.

### 3. 패키지 업데이트

설치된 패키지를 최신 버전으로 업데이트하려면 다음 명령어를 입력합니다:

```powershell
scoop update <패키지명>
```

예를 들어, Git을 업데이트하려면:

```powershell
scoop update git
```

### 4. 패키지 제거

설치된 패키지를 제거하려면 다음 명령어를 입력합니다:

```powershell
scoop uninstall <패키지명>
```

예를 들어, Git을 제거하려면:

```powershell
scoop uninstall git
```

### 5. Scoop 업데이트

Scoop 자체를 최신 버전으로 업데이트하려면 다음 명령어를 입력합니다:

```powershell
scoop update
```

이 명령어는 Scoop 자체와 모든 패키지들을 최신 버전으로 업데이트합니다.

## 참고자료

[[1] scoop 설치하기 (윈도우 명령어 설치 프로그램) - 네이버 블로그](https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=chandong83&logNo=221101838350)

[[2] Scoop 사용법(윈도우에서 설치 및 업데이트를 쉽게 할 수 있다)](https://velog.io/@nahyunbak/Scoop-%EC%82%AC%EC%9A%A9%EB%B2%95%EC%9C%88%EB%8F%84%EC%9A%B0%EC%97%90%EC%84%9C-%EC%84%A4%EC%B9%98-%EB%B0%8F-%EC%97%85%EB%8D%B0%EC%9D%B4%ED%8A%B8%EB%A5%BC-%EC%89%BD%EA%B2%8C-%ED%95%A0-%EC%88%98-%EC%9E%88%EB%8B%A4)

[[3] Scoop 사용법 - 이태원 블로그](https://leeted.tistory.com/221)

[[4] Scoop - dev_jubby.log](https://velog.io/@jubby/Scoop)