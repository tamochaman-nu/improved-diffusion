# improved-diffusion からの変更点まとめ

upstream: `openai/improved-diffusion` (base commit `1bc7bbb`)

単一ノード・単一GPU（RTX4090）・Docker(WSL2)環境で学習を回すために加えた変更。
`openai/guided-diffusion` は `improved-diffusion` から分岐したコードベースで、
`dist_util.py` / `train_util.py` / `fp16_util.py` 等の分散学習・チェックポイント
まわりの実装がほぼ同一なので、同じ問題が同じ形で再現するはず。

## 1. Attention を `F.scaled_dot_product_attention` に置き換え（速度）

`improved_diffusion/unet.py` の `QKVAttention.forward`。

```diff
 ch = qkv.shape[1] // 3
 q, k, v = th.split(qkv, ch, dim=1)
-scale = 1 / math.sqrt(math.sqrt(ch))
-weight = th.einsum(
-    "bct,bcs->bts", q * scale, k * scale
-)  # More stable with f16 than dividing afterwards
-weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
-return th.einsum("bts,bcs->bct", weight, v)
+q = q.transpose(-1, -2)
+k = k.transpose(-1, -2)
+v = v.transpose(-1, -2)
+out = F.scaled_dot_product_attention(q, k, v)
+return out.transpose(-1, -2)
```

- 手書きのeinsum+softmaxはTxTの注意行列を毎回materializeする。`F.scaled_dot_product_attention`
  （torch>=2.0）はFlash-Attention/memory-efficient kernelにディスパッチされ、
  `attention_resolutions`に32のような高解像度層（T=1024トークン）で特に効く。
- qkvはchannel-firstなので`transpose(-1, -2)`でsequence-firstに変換してから渡し、戻す。
- デフォルトの内部スケール`1/sqrt(ch)`は元コードの`scale*scale`（`1/sqrt(sqrt(ch))`を2回掛け）と等価なので、
  スケール引数を渡す必要はない。
- **guided-diffusionでの適用**: guided-diffusionは`resblock_updown`や`QKVAttentionLegacy`/
  `QKVAttention`の2系統があり、`use_new_attention_order`フラグでどちらを使うか切り替わる。
  該当する`forward`を両方とも同じ形で置き換える必要がある。

## 2. 単一プロセス時はDDPでラップしない（速度: 実測で1桁高速化）

`improved_diffusion/train_util.py` の `TrainLoop.__init__`。

```diff
-if th.cuda.is_available():
+if th.cuda.is_available() and dist.get_world_size() > 1:
     self.use_ddp = True
     self.ddp_model = DDP(...)
```

- `DDP`は勾配のバケット化・all-reduceのためにモデルパラメータへautograd hookを直接登録する。
  ラップした後で参照を差し替えても外せない。
- 同期する相手（他rank）が存在しない単一プロセス構成では、このhookは純粋なオーバーヘッドで、
  RTX4090 1枚で実測1ステップあたり約1桁遅くなっていた。
- `dist.get_world_size() > 1`のときだけDDPでラップするようにした。

## 3. 単一プロセス時はNCCLではなくgloo（安定性）

`improved_diffusion/dist_util.py` の `setup_dist`。

```diff
-backend = "gloo" if not th.cuda.is_available() else "nccl"
+backend = "gloo" if not th.cuda.is_available() or comm.size == 1 else "nccl"
```

- NCCLのP2P/共有メモリのトポロジ探索がDocker on WSL2環境でsegfaultすることが知られている。
- 単一プロセスならそもそも同期する相手がおらずNCCLの恩恵もないため、`comm.size == 1`のときは
  goloを使う。複数rank（実マルチGPU、`mpirun`経由）のときは従来通りNCCL。

## 4. `sync_params`: `p` ではなく `p.data` へbroadcast（バグ修正）

同じく `dist_util.py`。

```diff
 for p in params:
     with th.no_grad():
-        dist.broadcast(p, 0)
+        dist.broadcast(p.data, 0)
```

- goloバックエンドでパラメータテンソル自体（`requires_grad=True`のleaf Variable）へ
  in-placeで書き込むと、`no_grad()`の中でも "leaf Variable ... in-place operation" の
  チェックに引っかかってエラーになる。`.data`に書けばautograd管理外なので安全。

## 5. `cudnn.benchmark` / TF32 有効化（速度）

`dist_util.py` の `setup_dist` 冒頭に追加。

```python
if th.cuda.is_available():
    th.backends.cudnn.benchmark = True
    th.backends.cuda.matmul.allow_tf32 = True
    th.backends.cudnn.allow_tf32 = True
```

- 入力shape（image_size, batch_size）が毎ステップ固定なので、`cudnn.benchmark=True`で
  最速の畳み込みアルゴリズムを一度探索させてキャッシュさせる。
- TF32はAmpere/Ada世代のTensor Coreで行列積精度を少し犠牲にしてスループットを稼ぐ。
  学習には無視できる精度影響。

## 6. `load_state_dict`: 単一プロセス時はMPI bcastを経由しない（バグ修正・必須）

`dist_util.py` の `load_state_dict`。**guided-diffusionで再開機能を使うなら必ず当たる問題。**

```diff
 def load_state_dict(path, **kwargs):
+    comm = MPI.COMM_WORLD
+    if comm.Get_size() == 1:
+        with bf.BlobFile(path, "rb") as f:
+            data = f.read()
+        return th.load(io.BytesIO(data), **kwargs)
+
-    if MPI.COMM_WORLD.Get_rank() == 0:
+    if comm.Get_rank() == 0:
         with bf.BlobFile(path, "rb") as f:
             data = f.read()
     else:
         data = None
-    data = MPI.COMM_WORLD.bcast(data)
+    data = comm.bcast(data)
     return th.load(io.BytesIO(data), **kwargs)
```

- `mpi4py`のpickleベース`bcast`は**約2GiBを超えるメッセージで`MPI_ERR_ARG`になる既知の制限**がある。
- 256px・num_channels=256クラスのモデルだと、optimizerのcheckpoint（AdamWの1次・2次モーメント＋
  fp32マスターパラメータを含む）は本体重みの約2倍のサイズになり、本体（~1.9GiB）はギリギリ
  読めても`opt*.pt`（~3.7GiB）で確実に踏む。
- 単一プロセスなら自分自身にbroadcastする意味がないので、`comm.size==1`のときはMPIを経由せず
  直接読み込む。
- **注意**: モデル本体・EMAの読み込みは偶然2GiB未満で通ってしまうため、学習を最初から回している間は
  気づけない。**`--resume_checkpoint`で再開したときに初めて顕在化する**（今回もこのパターンで発覚）。
  guided-diffusionでも先に再開テストをしておくことを推奨。

## 7. DataLoaderの `num_workers: 1 → 0`

`improved_diffusion/image_datasets.py` の `load_data`。

- Docker on WSL2でのワーカープロセス起動まわりの安定性を優先して0にした変更（詳細な原因は未特定）。
- 実測ではバッチ取得に1ステップあたり約0.29秒（全体の約7%）を同期的に消費しており、
  `shm_size: '8gb'`を確保しているなら`num_workers=2`程度に戻せる余地がある。
  **これは意図的な高速化ではなく、まだ回収できていないコスト**として記録しておく。

## 8. 勾配累積 (`microbatch` < `batch_size`) 時のlr補正について（注意点）

`run_train.sh`のコメントで「`forward_backward()`がmicrobatch単位のlossを合計しており、
蓄積される勾配がbatch全体平均のK倍になる」問題（[openai/guided-diffusion#111](https://github.com/openai/guided-diffusion/issues/111)）
を踏まえ、`lr`を`基準lr / K`に補正して運用していた。

**ただし、これはAdamW（`weight_decay=0.0`）では実質no-opであることが判明した。**
Adamの更新量は `m_hat / (sqrt(v_hat) + eps)` で、勾配をK倍しても`m`と`sqrt(v)`が両方K倍
されるため比は変わらない。SGD/momentumのようにlrへ勾配が線形に効くオプティマイザでは
issue #111の指摘通りlr補正が必要だが、AdamWでは不要（むしろ意図せず実効lrを下げてしまう）。

- guided-diffusionもデフォルトはAdamWなので、同じ補正をしているなら見直す価値がある。
- weight_decay > 0で運用している場合は、decoupled weight decay項がlrに直接スケールするため
  影響の再検討が必要（多くの場合は無視できるほど小さいが、念のため）。

## 9. Docker化・`.env`によるデータパス外出し（環境整備、guided-diffusionには直接関係薄）

`Dockerfile` / `docker-compose.yml` / `.env.example` / `.gitignore` / `run_train.sh` を新規追加。
主要ポイントのみ:

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（torch>=2.1要）でRTX4090の24GB上限
  ギリギリでのアロケータ断片化による失速を緩和。
- ホスト側データセットパスは`.env`（`.gitignore`済み）の`HOST_DATA_DIR_ANIME`/`HOST_DATA_DIR_FFHQ`から
  `docker-compose.yml`経由で（`train-anime`/`train-ffhq`サービスそれぞれ）`/app/data`にマウントし、
  リポジトリにローカルパスを持ち込まない。
- `run_train.sh`は起動のたびに`pip install -e .`してから学習を開始する（`./:/app`がbind mountの
  ため、イメージビルド時点のコードと実行時のコードがズレないようにするため）。
- 学習再開の自動化: `$OPENAI_LOGDIR`内の`modelNNNNNN.pt`のうち最大ステップのものを検出して
  `--resume_checkpoint`に渡す（`train_util.py`側がoptimizer state・EMAも同じ命名規則で自動ロード
  してくれるため、これ以外の指定は不要）。

## guided-diffusionへ適用する際の優先順位

1. **#6（MPI bcastの2GiB制限）** — 再開機能を使うなら必須。踏むまで気づけないタイプのバグなので先に直しておく。
2. **#2（単一プロセスDDP回避）** — 単一GPU運用なら効果が一番大きい（実測1桁速い）。
3. **#1（SDPA化）** — 高解像度attention層があるモデルほど効く。
4. **#3, #4, #5** — 環境要因（WSL2/Docker）に依存する安定性・小型高速化。該当環境でなければ優先度は低い。
5. **#8** — lr補正ロジックを流用している場合は要見直し（AdamW運用なら外して基準lrに戻す）。
