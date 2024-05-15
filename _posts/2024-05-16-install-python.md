---
layout: single
title:  "Install python3 and jupyter lab in Windows"
categories: install
tag: [python, jupyter, pip, pipenv]
#toc: true
---

### Install Scoop

```bash
iex (new-object net.webclient).downloadstring('https://get.scoop.sh')
```
- Note: if you get an error you might need to change the execution policy (i.e. enable Powershell) with Set-ExecutionPolicy RemoteSigned -scope CurrentUser

[More Detail](https://github.com/ScoopInstaller/Install#readme)


### Install Python 3
```bash
scoop install python
```

### Upgrade pip
```bash
python3 -m pip install --upgrade pip
```

### Create directory structure for Jupyter Notebooks
- Note: feel free to change the path anything you want

```bash
mkdir C:\Projects\Jupyter\Notebooks
cd C:\Projects\Jupyter\Notebooks
```

### Install pipenv
```bash
python3 -m pip install pipenv
pipenv --python 3.7
```

### Install Jupyter Notebooks
```bash
pipenv install jupyterlab
```

### Install related ML libs
- Note: feel free to add/change any package you want or need

```bash
pipenv install pandas numpy matplotlib seaborn sklearn
```

### Run Jupyter Notebooks
- Note: you need the line below anytime you want to run Jupyter Notebooks

```bash
pipenv run jupyter lab
```

### Reference
[Ergin BULU - gist.github.com/ergin](https://gist.github.com/ergin)