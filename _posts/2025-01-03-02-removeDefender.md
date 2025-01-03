---
layout: single
title:  "Permanently disable Windows Defender Real time protection"
categories: tip
tag: [windows10, defender, safe mode]
toc: false
---

## boot safe mode
start logo -> power -> restart (with keep pressing shift) -> Problem Solving -> Advanced Option -> Problem Solving with Command line

## Registry add command for disable
```bash
REG ADD "HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Services\WinDefend" /v "DependOnService" /t REG_MULTI_SZ /d "RpcSs-DISABLED" /f >nul
REG ADD "HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Services\WinDefend" /v "Start" /t REG_DWORD /d "3" /f >nul
```

## Registry add command for enable
```bash
REG ADD "HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Services\WinDefend" /v "DependOnService" /t REG_MULTI_SZ /d "RpcSs" /f >nul
REG ADD "HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Services\WinDefend" /v "Start" /t REG_DWORD /d "2" /f >nul
```