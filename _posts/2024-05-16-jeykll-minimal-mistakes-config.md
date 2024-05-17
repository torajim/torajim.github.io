---
layout: single
title:  "Jeykll minimal mistake theme _config.yaml 필수 수정 사항"
categories: install
tag: [jeykll, blog, github]
toc: true
---

# 필수 수정 사항들
* locale: "ko-KR"
* url: myurl (ex., https://torajim.github.io)
* breadcrumbs: true
  * post에서 hierarchy 보여 주기 위함
* enable_copy_code_button: true
  * code block에서 C&C 할 수 있도록
* atom_feed.hide: true
  * RSS feed 제공하지 않을려면 끈다 (하단 footer가 너무 안이뻐서 아에 삭제함)
  * _includes > footer.html에서 하단 follow 부분을 전부 주석처리함 (너무 안이뻐서)
* search: true
  * blog에 검색 기능 추가
* Site Author
  * 이것저것 넣어 보면서 안이쁜건 다 뺐음
  * icon은 fontawesome.com 의 style을 사용함
  * fontawsome.com의 일부 icon들은 align이 안맞아서 무지성으로 추가하면 세로줄이 안맞아 안이쁨
    ```yaml
    # Site Author
    author:
      name             : # "Stefano Jang"
      avatar           : "/assets/images/zoozoobar_circle.png"
      bio              : # "동물사랑 캠퍼 개발자 아저씨"
      location         : # "South Korea"
      email            : # torajim@gmail.com
      links:
        - label: "Email"
          icon: "fas fa-fw fa-envelope-square"
          url: "mailto:torajim@gmail.com"
        - label: "Blogger"
          icon: "fas fa-fw fa-link"
          url: "https://www.zoozoobar.net"
        - label: "Google Scholar"
          icon: "fa-brands fa-google-scholar"
          url: "https://scholar.google.com/citations?user=YMAzgnYAAAAJ&hl=ko"
        - label: "GitHub"
          icon: "fab fa-fw fa-github"
          url: "https://github.com/torajim"
        - label: "Instagram"
          icon: "fab fa-fw fa-instagram"
          url: "https://instagram.com/apple2bapple"
        - label: "LinkedIn"
          icon: "fab fa-fw fa-linkedin"
          url: "https://www.linkedin.com/in/whjang"
    ```

# Local build 하고 실행시 발생하는 문제
* local에 ruby, msys2, ridk install후 bundle install 하면 추가한 md 파일을 개발환경에서 미리 볼 수 있다.
* 그런데 서버를 띄우고 나서 미리보기를 잘 한 이후, ctrl-c, 그 다음 다시 띄우려고 하면 아래와 같은 에러가 난다.

```bash
> bundle exec jekyll serve


[!] There was an error while loading `minimal-mistakes-jekyll.gemspec`: No such file or directory @ rb_sysopen - package.json. Bundler cannot continue.

 #  from C:/Users/toraj/workspace/torajim.github.io/_site/minimal-mistakes-jekyll.gemspec:3
 #  -------------------------------------------
 #    spec.add_development_dependency "rake", ">= 12.3.3"
 >  end
 #  require "json"
 #  -------------------------------------------

C:\Users\toraj\workspace\torajim.github.io>
```
* 이건 실행 시 _site라는 폴더가 생기고 여기 안에 있는 gemspec 파일이 package.json을 바라보지 못하면서 생기는 문제이다
* _site에 package.json을 옮겨주거나, _site를 수동으로 지우고 다시 서버를 띄우면 정상 동작한다
* 아래는 추가한 npm script (Windows 기준)
```js
  "scripts":{
    "start": "bundle exec jekyll serve",
    "clean": "rmdir /S/Q _site"
  }
```

* Jekyll build시 tzinfo 에러 문제
```powershell
C:/scoop/apps/ruby/3.3.1-1/lib/ruby/3.3.0/bundled_gems.rb:74:in `require': cannot load such file -- tzinfo (LoadError)
```
* tzinfo, tzinfo-data가 설치되었는지 확인
```powershell
gem install tzinfo
gem install tzinfo-data
```
* gemfile에 아래 두 줄을 추가한 후 jekyll build를 수행해 준다
```
gem 'tzinfo'
gem 'tzinfo-data', platforms: [:mingw, :mswin, :x64_mingw]
```