# Git

> Git은 분산형 버전관리(DVCS) 시스템 중 하나이다.

## 0. Git 기초 설정

* windows 환경에서는 `git for windows`로 검색하여 `git bash`를 설치한다.[다운로드 링크](https://gitforwindows.org/)

* 최초에 컴퓨터에서 git을 사용하는 경우 아래의 설정을 진행한다.

  ```bash
  $ git config --global user.email leejm456@gmail.com
  $ git config --global user.name Lee-JeongMin
  # 확인
  $ git config --global -l
  ```

  * 이메일 주소를 설정할 때, github에 가입된 이메일로 설정을 해야 커밋 이력이 github에 기록된다.

## 1. Git을 통한 버전관리 기본 흐름

### 1.1. Git 저장소 초기화

> 특정 폴더를 git 저장소로 활용하기 위해서 최초에 입력하는 명령어

```bash
$ git init
Initialized empty Git repository in C:/Users/student/Desktop/TIL/.git/
(master) $
```

* .git 폴더가 숨김 폴더로 생성되며, git bash에서는 (master) 라고 표기된다.
* 반드시 git으로 활용되고 있는 폴더 아래에서 저장소를 선언하지 말자.

### 1.2 . `add`

> 커밋 대상 파일들을 추가한다.

`add` 전 상황

``` bash
$ git status
On branch master

No commits yet
# 트랙킹 되지 않는 파일들
# => 새로 생성된 파일이고, git으로 관리 중이지 않는 파일
Untracked files:
	# git add 파일
	# 커밋이 될 것들을 포함시기키 위해서 위의 명령어를 써라!
  (use "git add <file>..." to include in what will be committed)
        git.md
        markdown-images/
        markdown.md

nothing added to commit but untracked files present (use "git add" to track)
```

`add` 후 상황

```bash
$ git add .
$ git status
```

```bash
On branch master

No commits yet
# 커밋될 변경사항들
Changes to be committed:
  (use "git rm --cached <file>..." to unstage)
        new file:   git.md
        new file:   "markdown-images/\354\272\241\354\262\230.PNG"
        new file:   markdown.md
```

* add 명령어는 아래와 같이 활용된다.

```bash
$ git add . # 현재 디렉토리 전부
$ git add git.md # 특정 파일
$ git add git.md markdown.md # 특정 파일 여러개
$ git add markdown-images/ # 특정 디렉토리
```

### 1.3. `commit`

> 이력을 확정 짓는 명령어

```bash
$ git commit -m '커밋메세지'
[master (root-commit) 28fb6b6] Init
 3 files changed, 87 insertions(+)
 create mode 100644 git.md
 create mode 100644 "markdown-images/\354\272\241\354\262\230.PNG"
 create mode 100644 markdown.md
 $ git status
 # 커밋할 것도 없고, 작업할 것도 없음
On branch master
nothing to commit, working tree clean
```

#### `log`

> 커밋 내역들을 확인할 수 있는 명령어

```bash
$ git log
Author: Lee-JeongMin <leejm456@gmail.com>
Date:   Wed Jul 8 14:41:51 2020 +0900

    Init

# 최근 n개 이력(1개)
$ git log -1
Author: Lee-JeongMin <leejm456@gmail.com>
Date:   Wed Jul 8 14:41:51 2020 +0900

    Init

# 간략한 표현
$ git log --oneline
28fb6b6 (HEAD -> master) Init

# 최근 n개 이력을 간략하게
$ git log --oneline -1
28fb6b6 (HEAD -> master) Init

```

## 2. 원격 저장소 활용

> 원격 저장소는(remote repository)를 제공하는 서비스는 많다.(gitlab, bitbucket)
>
> 그 중에서 github를 기준으로 설명하겠다.

### 2.1. 원격 저장소 등록

> git아! 원격 저장소(remote)로 등록해줘(add) origin이라는 이름으로 URL을!

```bash
$ git remote add origin 저장소url
```

* 저장소 확인

  ```bash
  $ git romote -v
  origin  https://github.com/Lee-JeongMin/TIL.git (fetch)
  origin  https://github.com/Lee-JeongMin/TIL.git (push)
  ```

* 저장소 삭제

  origin으로 지정된 저장소를 rm(remove)한다.

  ``` bash
  $ git remote rm origin
  ```

### 2.2. `push`

origin으로 설정된 원격저장소의 master 브랜치로 push한다.

```bash
$ git push origin master
```

### 2.3. `clone`

> 원격 저장소를 복제해온다.

```bash
~/집 $ git clone https://github.com/Lee-JeongMin/TIL.git
~/집 $ cd TIL
~/집/TIL (master) $
```

* 복제하는 경우 원격저장소 이름의 폴더가 생성된다.
* 해당 폴더로 이동하여 활용하면 된다.
* 이루 작업을 하는 경우 `add`, `commit`, `push`

### 2.4. `pull`

> 원격 저장소의 변경사항을 받아온다.

```bash
~/Desktop/TIL (master) $ git pull origin master
```



# 주의사항

> 원격 저장소와 로컬 저장소의 이력이 다르게 구성되는 경우
>
> 1) Github에서 직접 파일 수정을 하거나, 
>
> 2) 협업하는 과정이거나, 
>
> 3)집-강의장 환경으로 왔다갔다 하는 상황에서 발생할 수 있는 오류

```bash
$ git push origin master
To https://github.com/edutak/TIL--nlp.git
 ! [rejected]        master -> master (fetch first)
error: failed to push some refs to 'https://github.com/utak/TIL--nlp.git'
hint: Updates were rejected because the remote containsork that you do
hint: not have locally. This is usually caused by anoth repository pushing
hint: to the same ref. You may want to first integrate e remote changes
hint: (e.g., 'git pull ...') before pushing again.
hint: See the 'Note about fast-forwards' in 'git push -elp' for details.
```

* 이때, 원격 저장소 커밋 목록들과 로컬 저장소의 `git log --oneline`으로 비교해보면 다른점이 반드시 있을 것이다.

* 해결방법은 다음과 같다.

  1) 원격 저장소 내용을 받아온다.

  ```bash
  $ git pull origin master
  remote: Enumerating objects: 5, done.
  remote: Counting objects: 100% (5/5), done.
  remote: Compressing objects: 100% (3/3), done.
  remote: Total 3 (delta 2), reused 0 (delta 0), pack-reud 0
  Unpacking objects: 100% (3/3), 704 bytes | 7.00 KiB/s, ne.
  From https://github.com/edutak/TIL--nlp
   * branch            master     -> FETCH_HEAD
     173cf24..68ec3f5  master     -> origin/master
  Merge made by the 'recursive' strategy.
   "Github \355\231\234\354\232\251 \354\230\210\354\213\
   1 file changed, 1 insertion(+)
  ```

  * 이때 Vim 편집기 화면이 뜨는데, 커밋 메세지를 작성할 수 있는 곳이다.
    * `ese`를 누르고 `: wq`를 순서대로 입력한다.

  ![캡처dsf](markdown-images/%EC%BA%A1%EC%B2%98dsf.PNG)

  2) 다시 push를 한다.

  ```bash
  ![adf](../adf.PNG$ git log --oneline
  c30820c (HEAD -> master) Merge branch 'master' of https://github.com/edutak/TIL--nlp
  e151783 Add README
  68ec3f5 Update Github 활용 예시.md
  173cf24 Update files
  187ed91 Add clone command at home
  b523707 Update git.md
  6b6d274 Init
  $ git push origin master
  ```

  

  

