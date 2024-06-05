# 監視ツール

## はじめに

このツールは、GENIACの各種リソースの状態を監視するためのツールです。
ノードのダウンにより監視ができないという状態を回避するために環境外からの実行を想定しています。
※環境外のPCのダウンにより動かない状態は回避できませんが、現時点で設定しているマシンは極力落ちないように設定しています。

## 監視対象

- slurmのジョブの状態
- ディスク空き容量
- gitの更新状況
- ログインユーザの一覧

## ファイル説明

| ファイル名 | 説明                                                    | 実行場所 |
| ---------- | ------------------------------------------------------- | -------- |
| build.sh   | Dockerイメージを作成するスクリプト                      | 環境外PC |
| Dockerfile | Dockerイメージを作成するファイル                        | 環境外PC |
| exec.sh    | Dockerコンテナ内で実行されるスクリプト                  | Docker   |
| monitor.py | 取得したファイルを元にSlackへ通知を行うPythonスクリプト | Docker   |
| monitor.sh | 情報を取得するスクリプト。本番環境にて実行される        | 本番環境 |
| run.sh     | 環境外のPCからcronで起動されるスクリプト                | 環境外PC |

## 使い方

### Dockerのビルド

- Dockerコンテナを作成できる環境にて、以下のコマンドを実行します。

```bash
./build.sh
```

### Dockerの実行

- Dockerを実行できる環境にて、以下のコマンドを実行します。
  - なお、run.shは個別設定となりますので、以下の対応が必要です。
  - ./run.sh内でマウントしているボリュームの変更
  - gcloudのinitおよびsshが終了していること

```bash
./run.sh
```