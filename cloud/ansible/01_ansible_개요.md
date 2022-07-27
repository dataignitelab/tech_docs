# 다수 서버를 관리하기 위한 Ansible
![Ansible logo](../images/ansible_logo.png)

## 다수의 서버에서 시스템 환경 구성을 하려면?
빅데이터와 클라우드 환경에서는 다수의 서버를 활용하여 데이터를 저장, 처리, 서비스 하는 것이 필수적으로 요구됩니다. 대표적인 빅데이터 시스템이라고 하면 하둡이나 스파크, 카프카와 같은 것들이 떠오르는데, 다수의 컴퓨터들을 하나의 클러스터로 구성하여 대규모 데이터를 처리하게 됩니다. 만약 연구 또는 운영 하기 위하여 해당 시스템들을 클러스터로 구성한다고 했을때 일반적인 절차으로 VM(Virtual Machine)이나 컨테이너와 같은 기술들을 활용하여 최소 3대 이상의 서버를 준비하고 각각 환경 구성을 하기 위하여 Bash 쉘과 같은 환경에서 이런 저런 소프트웨어를 설치하게 됩니다. 이 때 리눅스 명령어들로 환경 구성을 한다면 일반적인 선택지는 다음 두개 일 겁니다.

* 선택1: 각 서버에 SSH 로 접근하여 하나하나 명령어들을 실행하여 필요 환경을 세팅한다.
* 선택2: 스마트하게 Shell script에 환경 구성 명령어들을 작성하고 실행한다.

선택1과 같이 각 서버들에 직접 명령어를 입력하는 방식은 매우 직관적이기 때문에 작업이 쉽겠지만 서버에 접속해야한다면 엄청난 반복작업이 필요하게 됩니다. 또한 선택2의 Shell Script 환경에서는 즉흥적인 명령어를 적용하기 위해서는 번거로운 작업이 필요할 뿐더러 shell script 에 대한 지식이 요구 됩니다.

`그러면 자주 사용하는 주요 기능을 미리 구성해놓고 서버 한 곳에서 실행하고 그에 대한 결과도 모니터링할 수 있다면 편하지 않을까요?`

그래서 등장하게 된 것이 프로비저닝 자동화 도구들입니다. 프로비저닝 자동화 도구는 다수의 서버들이 환경 세팅이나 명령어 입력 들을 한곳에서 관리하고 실행할 수 있는 도구입니다. 그 중에 대표적으로 Ansible, Terraform, Puppet, Check 같은 것들이 있습니다.

## Ansible : 프로비저닝 자동화 도구 
* Ansible은 다수의 리눅스 서버에서 시스템 관리 및 환경 구성을 지원하는 오픈소스 기반의 프로비저닝 자동화 도구 입니다.
* 다수의 서버의 환경 구성 및 배포를 코드로 정의하여 활용하는 `Infrastructure as a code`  의 개념에서 개발된 도구입니다.
* 2012년 오픈소스로 공개되었고 2015년 레드햇에 인수되어 현재는 클라우드 환경에서 DevOps의 핵심기술로서 다방면으로 활용되고 있습니다.

## Ansible 은 왜 쓰는걸까?
* 다수의 서버에서 다양한 환경 구성을 하기 위하여 서버에 따라 어떤 작업을 수행할 것인지에 대한 절차 및 방법을 정의하고 관리할 수 있습니다.
* 즉흥적인 명령어를 다수의 서버에 실행하고 결과를 확인할 수 있습니다.
* 설치가 매우 쉽습니다. SSH를 기반으로 동작하기 때문에, 관리 서버 1대에만 Ansible을 설치하고 나머지 서버에서는 설치할 필요가 없습니다. 물론! 모든 서버에서는 SSH 세팅과 서버 간 패스워드 없이 서로 명령어를 바로 실행할 수 있도록 public-key 공유와 리눅스 계정 설정은 미리 세팅해두어야 합니다.
* 멱등성(Idempotency)를 보장합니다. 동일한 설정을 여러번 실행해도 항상 동일한 결과를 보장합니다. 즉, 변경사항이 있을때만 실행한다고 보면 됩니다.
* 환경 구성에 대한 내용이 파일로 관리되므로 Git 으로 형상 관리가 가능합니다. 이를 기반으로 CI/CD 기능으로 확장이 가능하기 때문에 DevOps의 핵심요소기술 중 하나로 활용 가능합니다.

## Ansible 의 핵심 구성 요소
Ansible을 이해하기 위해서는 아래 3개의 요소에 대한 이해가 반드시 필요합니다.
* Inventory : 환경 구성을 실행할 서버들의 IP, Port, 사용자 계정, 서버 그룹 등을 정의
* Module : 서버에서 실행할 작업의 기본 단위
* Playbook : Inventory 에서 정의한 서버들에게 무슨 작업을 수행할 것인지 정의합니다. 작업의 단위로서 Module 들을 리스트를 작성하게됩니다.

### 1. Inventory
환경 구성을 실행할 서버들의 IP, Port, 사용자계정, 서버 그룹을 정의하는 파일입니다.
```bash
# Inventroy.ini 예시

master01 ansible_host=172.17.0.1 ansible_port=1022 ansible_user=test_user
master02 ansible_host=172.17.0.2 ansible_port=1022 ansible_user=test_user
worker01 ansible_host=172.17.0.3 ansible_port=1022 ansible_user=test_user
worker02 ansible_host=172.17.0.4 ansible_port=1022 ansible_user=test_user
worker03 ansible_host=172.17.0.5 ansible_port=1022 ansible_user=test_user

[masters]
master01
master02

[workers]
worker01
worker02
worker03
```

master01, worker01 은 Ansible 에서 사용할 서버의 이름(변수)이며, 대괄호로 묶여진 것은 서버의 그룹명입니다. 그룹 단위로 서버 관리가 가능합니다. 주요 설정 항목은 아래와 같습니다.
* ansible_host : 서버의 IP 또는 호스트명
* ansible_port : 서버의 SSH Port
* ansible_user : 명령이 실행될 해당 서버의 리눅스 계정명

### 2. Module
Playbook 에서 수행할 작업의 단위입니다. 어떤 작업을 수행할 것인지 Ansible 에서 미리 정의한 기능들이라고 이해하면 됩니다. 
Module 은 Ansible 의 핵심이므로 어떤 기능들이 있는지? 어떤 방식으로 사용하는지? 익히는 것이 중요합니다. 현재 대략 500여개의 기능들이 제공되고 있으며, 일반적인 리눅스 명령어들과 퍼블릭 클라우드 서비스들의 명령어를 지원합니다.
주요한 명령어 카테고리는 아래와 같습니다. ([Module Index — Ansible Documentation](https://docs.ansible.com/ansible/2.9/modules/modules_by_category.html))
* [Cloud modules](https://docs.ansible.com/ansible/2.9/modules/list_of_cloud_modules.html)
* [Clustering modules](https://docs.ansible.com/ansible/2.9/modules/list_of_clustering_modules.html)
* [Commands modules](https://docs.ansible.com/ansible/2.9/modules/list_of_commands_modules.html)
* [Crypto modules](https://docs.ansible.com/ansible/2.9/modules/list_of_crypto_modules.html)
* [Database modules](https://docs.ansible.com/ansible/2.9/modules/list_of_database_modules.html)
* [Files modules](https://docs.ansible.com/ansible/2.9/modules/list_of_files_modules.html)
* [Identity modules](https://docs.ansible.com/ansible/2.9/modules/list_of_identity_modules.html)
* [Inventory modules](https://docs.ansible.com/ansible/2.9/modules/list_of_inventory_modules.html)
* [Messaging modules](https://docs.ansible.com/ansible/2.9/modules/list_of_messaging_modules.html)
* [Monitoring modules](https://docs.ansible.com/ansible/2.9/modules/list_of_monitoring_modules.html)
* [Net Tools modules](https://docs.ansible.com/ansible/2.9/modules/list_of_net_tools_modules.html)
* [Network modules](https://docs.ansible.com/ansible/2.9/modules/list_of_network_modules.html)
* [Notification modules](https://docs.ansible.com/ansible/2.9/modules/list_of_notification_modules.html)
* [Packaging modules](https://docs.ansible.com/ansible/2.9/modules/list_of_packaging_modules.html)
* [Remote Management modules](https://docs.ansible.com/ansible/2.9/modules/list_of_remote_management_modules.html)
* [Source Control modules](https://docs.ansible.com/ansible/2.9/modules/list_of_source_control_modules.html)
* [Storage modules](https://docs.ansible.com/ansible/2.9/modules/list_of_storage_modules.html)
* [System modules](https://docs.ansible.com/ansible/2.9/modules/list_of_system_modules.html)
* [Utilities modules](https://docs.ansible.com/ansible/2.9/modules/list_of_utilities_modules.html)
* [Web Infrastructure modules](https://docs.ansible.com/ansible/2.9/modules/list_of_web_infrastructure_modules.html)
* [Windows modules](https://docs.ansible.com/ansible/2.9/modules/list_of_windows_modules.html)

### 3. Playbook
Inventory 에서 정의한 서버들에게 무슨 작업을 수행할 것인지 정의하는 YAML 파일입니다. 어떤 서버에서 어떤 Module을 실행할 것인지 정의하게 됩니다.  위에서도 설명한 것처럼 Module 로 정의된 기능만 활용 가능하며 기본적인 리눅스 명령어들은 대부분 Module 로 제공하고 있습니다.

```yaml
# Playbook.yaml 파일 작성 예시
# 도커 설치하는 명령어를 예시로 작성하였습니다.

---
- name: install
  hosts: all
  become: true
  tasks:
    - apt_key:
        url: https://download.docker.com/linux/ubuntu/gpg
        state: present
    - apt_repository:
        repo: deb [arch=amd64] https://download.docker.com/linux/ubuntu focal stable
        state: present
    - name: 도커 설치
      apt:
        name: docker-ce
        state: latest
        update_cache: yes        
    - name: 도커 data 폴더 위치 변경을 위한 daemon.json 파일 생성
      copy:
        dest: /etc/docker/daemon.json
        content: |
          {
            "exec-opts": ["native.cgroupdriver=systemd"],
            "data-root": "/home/data/docker"
          }
    - name: 사용자 계정 docker 그룹에 추가
      user:
        name: test_user
        groups: docker
        append: yes
    - name: 도커 실행
      shell: 'systemctl start docker.service'
```

주요 항목 설명은 아래와 같습니다.
* hosts : 실행할 그룹명입니다. all 으로 지정한다면 전체에게 실행하겠다는 의미입니다.
* become : root 로 실행하겠다는 의미입니다.
* task : 실행할 작업의 목록입니다.
* apt_key, apt_repository : apt 저장소를 추가하는 모듈입니다.
* apt : 우분투에서 사용하는 패키지 설치 명령어 기능을 활용하는 모듈입니다.
* copy : content 에 명시된 내용으로 파일을 생성합니다.
* user : 유저 권한에 관련도닏 명령어에 대한 모듈입니다.
* shell : 리눅스 Shell 명령어를 직업 입력하는 모듈입니다.

## 스마트한 클러스터 관리를 위한 Ansible 
빅데이터, AI, IoT, 클라우드를 융합하는 것이 기본이 되는 요즘과 같은 시대에서는 대규모의 클러스터를 운영하는 것이 필수적으로 요구되고 있습니다.  대규모의 클러스터 환경을 구성한다는 것은 너무나도 많은 시간을 소비하는 작업입니다. 그러한 대규모 클러스터 환경에서 Ansible를 활용한다면 코드 기반의 서비스 배포 및 관리를 수행함으로서 많은 이점을 누릴수 있게 됩니다. 단순하게 개인 연구 목적으로 작은 클러스터를 운영한다면 굳이 필요 없을 수도 있겠으나 언젠가는 대규모의 클러스터를 운영하는 순간이 온다면 Ansible와 같은 프로비저닝 툴들의 도입을 검토해보시길 바랍니다.
