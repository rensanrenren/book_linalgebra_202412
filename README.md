# book_linalgebra_202412
# Pythonで学ぶ線形代数学

# 準備
```plaintext

## 本書の目的

この本では、Pythonを使いながら線形代数学を実務に応用できるスキルを身につけることを目的としています。そのために、まず必要な環境を整え、基本的なツールやライブラリの使い方を学ぶところから始めます。本章では、学習を始める前に行うべき準備について解説します。

---

### 1. 必要な環境とツール

本書を進めるにあたり、以下の環境やツールを準備してください。

#### ハードウェア
- コンピュータ（Windows, macOS, またはRaspberry Pi）
  - 推奨スペック: メモリ4GB以上、空きストレージ10GB以上

#### ソフトウェア
- Python 3.8以降（本書では3.10を推奨）
- 必須ライブラリ: NumPy, Matplotlib, SciPy, VPython
- その他のツール: Jupyter Notebook, LaTeX（TeX LiveまたはMiKTeX）

#### 推奨するエディタ
- Visual Studio Code（拡張機能: Python, Jupyter）

---

### 2. Pythonのインストールと環境構築

Pythonをインストールして、環境をセットアップします。

#### WindowsまたはmacOSの場合
1. [Python公式サイト](https://www.python.org/downloads/)にアクセスし、最新の安定版をダウンロードしてください。
2. インストーラーを実行し、以下の設定を行います。
   - 「Add Python to PATH」にチェックを入れる（Windows）。
   - カスタムインストールでpipとIDLEをインストールする。

#### Raspberry Piの場合
Raspberry Pi OSにはPythonがプリインストールされています。ただし、最新バージョンが必要な場合は以下を実行してください。

```bash
sudo apt update
sudo apt install python3 python3-pip
```

---

### 3. 必須ライブラリのインストール

以下のコマンドをターミナルまたはコマンドプロンプトで実行し、ライブラリをインストールしてください。

```bash
pip install numpy matplotlib scipy vpython jupyter
```

インストールが完了したら、以下のコマンドでインストール済みのライブラリを確認できます。

```bash
pip list
```

---

### 4. Jupyter Notebookのセットアップ

Jupyter Notebookは、Pythonコードを記述・実行しながら説明文や図を加えられる便利なツールです。

#### インストール手順
以下のコマンドを実行してください。

```bash
pip install notebook
```

インストールが完了したら、以下のコマンドで起動します。

```bash
jupyter notebook
```

ブラウザに表示されるホーム画面で新規ノートブックを作成し、以下のコードを入力して動作確認を行いましょう。

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

plt.plot(x, y)
plt.title("Sine Wave")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.show()
```

---

### 5. その他のツールの準備

#### LaTeX
数式を美しく表現するためにLaTeXを利用します。以下のいずれかのディストリビューションをインストールしてください。
- [TeX Live](https://www.tug.org/texlive/)
- [MiKTeX](https://miktex.org/)

インストール後、数式の動作確認を行います。例えば、以下のコードをLaTeXでコンパイルしてみてください。

```latex
\documentclass{article}
\usepackage{amsmath}
\begin{document}
\[ a^2 + b^2 = c^2 \]
\end{document}
```

#### PyX
PyXを使って線形代数学の概念を視覚化する準備をします。

以下のコマンドでインストールします。

```bash
pip install pyx
```

PyXが正しく動作するか、以下のコードを実行して確認してください。

```python
from pyx import canvas, path

c = canvas.canvas()
c.stroke(path.line(0, 0, 2, 2))
c.writePDFfile("example")
```

---

### 6. トラブルシューティング

#### ライブラリのインストールに失敗する場合
- pipのバージョンが古い場合があります。以下のコマンドでアップグレードしてください。

```bash
pip install --upgrade pip
```

#### Jupyter Notebookが起動しない場合
- インストールが正しく行われているか確認します。
- 以下のコマンドでJupyterがインストール済みか確認してください。

```bash
pip show notebook
```



これで準備が完了しました。次章では、Pythonの基本操作について学びます。本書の内容を最大限に活用するために、設定した環境をぜひ確認してみてください。

# 第0章 Pythonの基本操作

この章では、Pythonの基本操作を学びます。Raspberry PiへのPythonのインストールから、基本構文やライブラリの利用方法、Jupyter NotebookやVPythonの使い方を段階的に解説します。本書を進めるうえで必要な基礎を築くことが目的です。

---

## 03 Raspberry Piへのインストール

### Raspberry Piとは
Raspberry Pi（ラズベリーパイ）は、手のひらサイズのシングルボードコンピュータです。低価格で入手可能でありながら、通常のコンピュータと同様の機能を持ち、Pythonプログラミングの学習やIoTデバイスの開発など幅広い用途に利用されています。

#### 必要な準備
Raspberry PiでPythonを利用するための手順を以下に示します。

#### 手順
1. Raspberry Pi OSを最新に更新します。
   ```bash
   sudo apt update
   sudo apt upgrade
   ```

2. Python 3がインストールされているか確認します。
   ```bash
   python3 --version
   ```

3. 必要なライブラリをインストールします。
   ```bash
   sudo apt install python3-pip python3-venv
   ```

4. pipのバージョンを確認します。
   ```bash
   pip3 --version
   ```

これでPythonの環境が整いました。

---

## 04 Pythonの起動

Pythonを起動し、簡単な操作を試してみます。

### インタープリタの起動
ターミナルまたはコマンドプロンプトで以下を実行します。
```bash
python3
```

インタラクティブシェルが起動し、以下のようなプロンプトが表示されます。

```plaintext
Python 3.x.x (default, ...)
Type "help", "copyright", "credits" or "license" for more information.
>>>
```

### 簡単な計算を試す
```python
>>> 2 + 3
5
>>> print("Hello, Python!")
Hello, Python!
```

終了するには以下を入力します。
```plaintext
>>> exit()
```

---

## 05 ライブラリの利用

Pythonにはさまざまなライブラリがあり、本書では以下のライブラリを使用します。

### NumPyとは
NumPy（Numerical Python）は、Pythonの数値計算ライブラリです。多次元配列の操作や数学的計算（線形代数、統計、フーリエ変換など）に特化しています。本書で扱う行列演算のほとんどは、NumPyを使って実装します。

#### NumPyの基本操作
```python
import numpy as np

array = np.array([1, 2, 3])
print("配列:", array)
```

---

### Matplotlibとは
Matplotlibは、Pythonでグラフや図を描画するためのライブラリです。簡単なプロットから高度な可視化まで幅広く対応しています。

#### Matplotlibでのグラフ描画
```python
import matplotlib.pyplot as plt

x = [1, 2, 3]
y = [2, 4, 6]

plt.plot(x, y)
plt.title("グラフの例")
plt.show()
```

---

## 06 VPythonの利用

### VPythonとは
VPythonは、Pythonで3Dオブジェクトを簡単に描画できるライブラリです。物理シミュレーションや3D可視化の学習に適しています。

#### サンプルコード
```python
from vpython import sphere, vector

sphere(pos=vector(1, 2, 3), radius=1, color=vector(1, 0, 0))
```

実行するとブラウザに3Dオブジェクトが表示されます。

---

## 07 Pythonの構文

Pythonの基本構文を紹介します。

### 変数と演算
```python
x = 10
y = 20
print("合計:", x + y)
```

### 条件分岐
```python
if x > y:
    print("xはyより大きい")
else:
    print("xはy以下")
```

### ループ
```python
for i in range(5):
    print(i)
```

---

## 08 インポート

Pythonでは外部モジュールやライブラリをインポートして活用できます。

### 基本のインポート
```python
import math

print("円周率:", math.pi)
```

---

## 09 Jupyter Notebookの利用

### Jupyter Notebookの概要
Jupyter Notebookは、Pythonコードの記述、実行、結果の記録を一つのノートブック形式で行えるツールです。データ分析や機械学習の分野で特に人気があります。

#### サンプルコード
以下のコードを実行して動作を確認しましょう。
```python
print("Hello, Jupyter!")
```

---

## 010 LaTeXなどその他のツール

### LaTeXやPyXとは
- **LaTeX**: 数式や文章を美しく組版するためのツール。学術文書の作成に広く利用されています。
- **PyX**: Pythonで高品質な図やグラフを描画するライブラリ。本書では線形代数学の可視化に使用します。

### LaTeXで数式を記述
以下をLaTeXエディタでコンパイルします。
```latex
\documentclass{article}
\usepackage{amsmath}
\begin{document}
\[ a^2 + b^2 = c^2 \]
\end{document}
```

---

## 011 Pythonの32ビット版と64ビット版について

### Pythonの32ビット版と64ビット版の違い
- **32ビット版**: メモリ使用量が少なく、古いシステムでも動作する。ただし、使用できるメモリ量に制限があり、大規模なデータ処理には不向き。
- **64ビット版**: メモリ制限が緩和され、大規模なデータ処理が可能。本書では64ビット版を推奨します。

#### 確認方法
以下のコマンドでインストールされているPythonのビット数を確認できます。
```bash
python3 -c "import platform; print(platform.architecture())"
```

---

これで「第0章 Pythonの基本操作」の説明は終了です。この章を参考にPythonの基礎を習得し、次の章へ進んでください。

以下に「第1章 数学の基礎とPythonによる表現」の完成版を提示します。

---

# 第1章 数学の基礎とPythonによる表現

この章では、数学の基礎をPythonを使って表現する方法を学びます。実数や複素数、集合、順序対、写像などの数学的概念をPythonでどのように実装するかを解説し、Pythonのクラスやオブジェクト、リストや行列の操作にも触れます。また、画像データの処理やPyXを用いた数学的描画についても学びます。

---

## 12 実数と複素数

### 実数とは
実数は、整数、小数、無理数（例: √2）を含むすべての数を指します。Pythonでは、`int`型（整数）や`float`型（浮動小数点数）で表現されます。

#### 実数の基本操作
```python
# 整数と浮動小数点数
a = 5
b = 3.14

# 演算
print("加算:", a + b)
print("累乗:", a ** 2)
```

### 複素数とは
複素数は、実部と虚部を持つ数です。Pythonでは`complex`型で扱われ、`j`で虚数単位を表します。

#### 複素数の基本操作
```python
z = 3 + 4j  # 実部: 3, 虚部: 4
print("複素数:", z)
print("共役:", z.conjugate())
print("絶対値:", abs(z))
```

---

## 13 集合

### 集合とは
集合は、重複しない要素の集まりで、Pythonでは`set`型で表現されます。

#### 集合の基本操作
```python
A = {1, 2, 3}
B = {3, 4, 5}

# 和集合
print("和集合:", A | B)
# 積集合
print("積集合:", A & B)
# 差集合
print("差集合:", A - B)
```

---

## 14 順序対とタプル

### 順序対とは
順序対は、要素が順序付けられた2つの値の組で、Pythonでは`tuple`型で表現されます。

#### タプルの例
```python
pair = (1, 2)
print("順序対:", pair)
print("1番目の要素:", pair[0])
```

---

## 15 写像と関数

### 写像とは
写像は、集合の要素を別の集合の要素に対応させるルールです。Pythonでは関数として表現されます。

#### 関数の例
```python
def f(x):
    return x ** 2

print("f(3):", f(3))
```

---

## 16 Pythonにおけるクラスとオブジェクト

### クラスとオブジェクトとは
クラスはオブジェクトの設計図で、オブジェクトはその設計図から作られた具体的な実体です。

#### クラスの基本
```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def move(self, dx, dy):
        self.x += dx
        self.y += dy

p = Point(1, 2)
p.move(3, 4)
print("新しい座標:", p.x, p.y)
```

---

## 17 リスト配列および行列

### リストとは
リストは、順序付けられた要素の集まりで、Pythonでは`list`型を使用します。

#### リストの基本操作
```python
lst = [1, 2, 3]
print("リスト:", lst)
lst.append(4)
print("追加後:", lst)
```

### NumPyによる行列操作
```python
import numpy as np

matrix = np.array([[1, 2], [3, 4]])
print("行列:")
print(matrix)
```

---

## 18 画像データの準備

### 画像データの読み込み
画像データは、数値の行列として扱うことができます。

#### Pythonでの画像操作
```python
from PIL import Image
import numpy as np

image = Image.open("example.png").convert("L")
image_array = np.array(image)
print("画像データの行列:")
print(image_array)
```

---

### 182 複素数値化した手書き文字データをGUIで作る

#### 実装例
GUIライブラリを使用して手書き文字データを複素数値化する方法を解説します。

```python
# GUIサンプル（tkinter使用例）
import tkinter as tk

def draw():
    pass  # GUIで手書きしたデータを複素数値化する処理をここに記述

root = tk.Tk()
root.mainloop()
```

---

### 183 グレースケールの手書き文字データ

#### グレースケール画像の処理
Pythonでグレースケール画像を操作する例を示します。

```python
gray_image = image.convert("L")
gray_array = np.array(gray_image)
```

---

## PyX

### PyXの概要
PyXは、高品質な図やグラフをPythonで描画するライブラリで、数学的描画や可視化に特化しています。

#### PyXの使用例
```python
from pyx import canvas, path

c = canvas.canvas()
c.stroke(path.line(0, 0, 2, 2))
c.writePDFfile("example")
```

以下に「第2章 線形空間と線形写像」の完成版を提示します。

---

# 第2章 線形空間と線形写像

この章では、線形空間と線形写像について学びます。部分空間や線形写像の基本的性質を理解し、Pythonを用いてそれらを表現する方法を解説します。また、「音を見る」というセクションでは、音声データの可視化を通じて線形代数学の応用例を学びます。

---

## 22 部分空間

### 部分空間とは
部分空間は、線形空間 \( V \) の部分集合で、以下の条件を満たすものを指します：
1. 零ベクトルを含む。
2. ベクトルの加法について閉じている。
3. スカラー倍について閉じている。

例えば、3次元空間 \( \mathbb{R}^3 \) の中の平面 \( ax + by + cz = 0 \) は部分空間です。

---

### Pythonで部分空間を表現する
PythonではNumPyを使って部分空間の操作を表現できます。

#### Pythonコード例
```python
import numpy as np

# ベクトル集合
v1 = np.array([1, 2, 3])
v2 = np.array([2, 4, 6])

# 線形結合
a, b = 2, -1
v_comb = a * v1 + b * v2
print("線形結合:", v_comb)

# 零ベクトルの確認
print("零ベクトル:", np.allclose(v_comb, np.zeros_like(v_comb)))
```

---

## 23 線形写像

### 線形写像とは
線形写像は、線形空間 \( V \) から \( W \) への関数 \( f \) であり、以下の条件を満たします：
1. 加法について線形性を持つ：\( f(u + v) = f(u) + f(v) \)
2. スカラー倍について線形性を持つ：\( f(cu) = c f(u) \)

例えば、2次元平面でベクトルを回転させる操作は線形写像です。

---

### Pythonで線形写像を実装する
線形写像は行列として表現できます。PythonではNumPyの`dot`関数を使って実装します。

#### Pythonコード例
```python
import numpy as np

# 回転行列（45度回転）
theta = np.pi / 4
rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta),  np.cos(theta)]])

# ベクトル
v = np.array([1, 0])

# 線形写像
v_rotated = np.dot(rotation_matrix, v)
print("回転後のベクトル:", v_rotated)
```

---

## 音を見る

### 音と線形代数学
音は時間領域での波として表現されますが、線形代数学を使うことで周波数領域に変換できます。この変換により、音の特徴を視覚化したり解析したりすることが可能です。

---

### 音声データを可視化する例
PythonのSciPyとMatplotlibを使用して、音声データを時間領域と周波数領域で視覚化する例を示します。

#### Pythonコード例
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# サンプルデータ: サイン波
fs = 500  # サンプリング周波数
t = np.linspace(0, 1, fs, endpoint=False)  # 時間
freq = 5  # 周波数5Hz
signal = np.sin(2 * np.pi * freq * t)

# フーリエ変換
fft_signal = fft(signal)
frequencies = fftfreq(len(t), d=1/fs)

# 結果をプロット
plt.figure(figsize=(12, 6))

# 時間領域
plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title("時間領域")
plt.xlabel("時間 [秒]")
plt.ylabel("振幅")

# 周波数領域
plt.subplot(2, 1, 2)
plt.plot(frequencies[:fs//2], np.abs(fft_signal)[:fs//2])
plt.title("周波数領域")
plt.xlabel("周波数 [Hz]")
plt.ylabel("振幅")

plt.tight_layout()
plt.show()
```

---

### 実行結果の解説
- **時間領域**では、波形がどのように変化しているかを観察できます。
- **周波数領域**では、波形を構成する周波数成分が確認できます。この例では、5Hzの成分がピークとなるグラフが表示されます。

以下に、読みやすさを意識して改善した「第3章 基底と次元」の完成版を提示します。

---

# 第3章 基底と次元

この章では、線形代数学の重要な概念である基底と次元について学びます。まず、線形独立と線形従属の違いを直感的に理解し、具体例とPythonコードを用いて確認します。次に、基底の定義とその役割を詳しく説明し、行列の階数とその応用についても扱います。最後に、次元の注意点を述べ、実務での活用方法に触れます。

---

## 32 線形独立と線形従属

### 線形独立とは
複数のベクトルが **線形独立** であるとは、次の条件を満たすことを意味します：
- ベクトルの線形結合が零ベクトル（全成分が0のベクトル）となる場合、その係数がすべて0でなければならない。

#### 数式で表現すると
\[
c_1v_1 + c_2v_2 + \cdots + c_nv_n = 0
\]
が成り立つとき、\( c_1 = c_2 = \cdots = c_n = 0 \) が唯一の解である場合、ベクトル \( v_1, v_2, \ldots, v_n \) は線形独立です。

#### 視覚的な理解
- 2次元空間では、平行でない2つのベクトルは線形独立です。
- 3次元空間では、同一平面上にない3つのベクトルは線形独立です。

---

### 線形従属とは
複数のベクトルが **線形従属** であるとは、次の条件を満たすことを意味します：
- ベクトルの線形結合が零ベクトルとなる場合に、少なくとも1つの係数が0以外である。

#### 数式で表現すると
\[
c_1v_1 + c_2v_2 + \cdots + c_nv_n = 0
\]
が成り立つとき、\( c_1, c_2, \ldots, c_n \) の中に0でない係数が存在する場合、ベクトル \( v_1, v_2, \ldots, v_n \) は線形従属です。

#### 視覚的な理解
- 2次元空間では、平行な2つのベクトルは線形従属です。
- 3次元空間では、同一平面上に存在する3つのベクトルは線形従属です。

---

### Pythonで線形独立性を確認する
NumPyを使って、ベクトルの線形独立性を判定します。

#### Pythonコード例
```python
import numpy as np

# ベクトル集合
v1 = np.array([1, 2, 3])
v2 = np.array([2, 4, 6])
v3 = np.array([3, 6, 9])

# 行列にベクトルを配置
matrix = np.column_stack([v1, v2, v3])

# 行列のランクを計算
rank = np.linalg.matrix_rank(matrix)
print("行列のランク:", rank)

# ベクトルの数
num_vectors = matrix.shape[1]

# 線形独立の判定
if rank == num_vectors:
    print("ベクトルは線形独立です")
else:
    print("ベクトルは線形従属です")
```

---

## 33 基底と表現

### 基底とは
基底とは、次の2つの条件を満たすベクトルの集合です：
1. 線形空間を張る（任意のベクトルが基底ベクトルの線形結合で表現できる）。
2. 基底ベクトルは線形独立である。

#### 基底の直感的な理解
- 平面 \( \mathbb{R}^2 \) の標準基底は、\( \{(1, 0), (0, 1)\} \) です。この2つのベクトルを用いれば、平面上の任意のベクトルを表現できます。
- 空間 \( \mathbb{R}^3 \) の標準基底は、\( \{(1, 0, 0), (0, 1, 0), (0, 0, 1)\} \) です。

#### 数式での表現
任意のベクトル \( v \) を基底 \( \{b_1, b_2, \ldots, b_n\} \) を用いて表すと、
\[
v = c_1b_1 + c_2b_2 + \cdots + c_nb_n
\]
の形になります。

---

### Pythonで基底を用いたベクトルの表現
NumPyを使って基底を用いたベクトルの表現を計算します。

#### Pythonコード例
```python
import numpy as np

# 基底ベクトル
basis = np.array([[1, 0], [0, 1]])

# ベクトル
v = np.array([3, 4])

# 基底での表現
coefficients = np.linalg.solve(basis, v)
print("基底での表現:", coefficients)
```

---

## 34 次元と階数

### 次元とは
次元は、線形空間を張るために必要な最小の基底ベクトルの数を指します。
- 平面（2次元空間）は次元が2。
- 空間（3次元空間）は次元が3。

#### 次元の具体例
- 1次元の線は、1つの基底ベクトルで張ることができます。
- 2次元の平面は、2つの線形独立なベクトルで張ることができます。

---

### 行列の階数とは
行列の階数（ランク）は、行または列の線形独立なベクトルの最大数を意味します。

#### 階数の解釈
行列の階数は次のことを表します：
- 行列が表現できる線形空間の次元。
- 行または列ベクトルの独立性。

---

### Pythonで階数を確認する
行列の階数をNumPyで計算する方法を示します。

#### Pythonコード例
```python
import numpy as np

# 行列
matrix = np.array([[1, 2, 3],
                   [2, 4, 6],
                   [3, 6, 9]])

# 行列の階数
rank = np.linalg.matrix_rank(matrix)
print("行列の階数:", rank)
```

#### 実行結果
上記の行列の階数は1であり、この行列が1次元の線形空間を表現していることを示します。

---

## 次元に関する注意

### 次元の応用と解釈
次元は、抽象的な数学の概念としてだけでなく、実際の応用でも重要です。
- **データ分析**: データの次元は特徴量の数を指します。次元削減は計算コストを下げるために重要です。
- **関数空間**: 次元は、フーリエ変換や微分方程式の解空間で使われます。

以下に「第4章 行列」の完成版を提示します。

---

# 第4章 行列

この章では、行列の基本的な概念から線形写像との関係、行列の積や逆行列、基底変換、相似変換、随伴行列に至るまでを学びます。また、行列計算の効率性についても触れ、Pythonを使った具体的な計算例を通じて理解を深めます。

---

## 行列と線形写像

### 行列とは
行列は、数を縦と横に並べた二次元の配列です。行列は次のように表されます：
\[
A = \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix}
\]
行列は、線形写像（線形変換）を表す重要な道具です。

---

### 線形写像と行列の関係
線形写像 \( f: \mathbb{R}^n \to \mathbb{R}^m \) は、対応する行列 \( A \) を使って次のように表現されます：
\[
f(x) = A \cdot x
\]
ここで、\( x \) はベクトル、\( A \) は行列です。これにより、ベクトルの変換操作が簡潔に記述できます。

---

### Pythonで行列を使った線形写像
NumPyを使って、行列を操作し線形写像を実現します。

#### Pythonコード例
```python
import numpy as np

# 行列
A = np.array([[2, 1], 
              [1, 3]])

# ベクトル
x = np.array([1, 2])

# 線形写像
y = np.dot(A, x)
print("線形写像の結果:", y)
```

---

## 43 線形写像の合成と行列の積

### 行列の積とは
行列 \( A \) と \( B \) の積 \( C = A \cdot B \) は、以下のように計算されます：
- \( C \) の要素 \( c_{ij} \) は、行列 \( A \) の第 \( i \) 行と行列 \( B \) の第 \( j \) 列の内積です。

#### 数式
\[
c_{ij} = \sum_{k=1}^n a_{ik} \cdot b_{kj}
\]

---

### 線形写像の合成と行列の積
2つの線形写像 \( f \) と \( g \) を合成した写像 \( h = f \circ g \) は、対応する行列の積で表現されます：
\[
h(x) = A \cdot (B \cdot x) = (A \cdot B) \cdot x
\]

---

### Pythonで行列の積を計算する
#### Pythonコード例
```python
import numpy as np

# 行列 A と B
A = np.array([[1, 2], 
              [3, 4]])
B = np.array([[2, 0], 
              [1, 2]])

# 行列の積
C = np.dot(A, B)
print("行列の積:\n", C)
```

---

## 逆行列と基底の変換、行列の相似

### 逆行列とは
行列 \( A \) の逆行列 \( A^{-1} \) は、次の性質を満たします：
\[
A \cdot A^{-1} = A^{-1} \cdot A = I
\]
ここで、\( I \) は単位行列です。

#### 注意
逆行列は、正則行列（ランクがフルの行列）にのみ存在します。

---

### 基底の変換と行列の相似
基底を変換する際、行列 \( A \) は変換行列 \( P \) を使って次のように相似変換されます：
\[
B = P^{-1} \cdot A \cdot P
\]
ここで、行列 \( B \) は新しい基底での表現です。

---

### Pythonで逆行列と相似変換を計算する
#### Pythonコード例
```python
import numpy as np

# 行列
A = np.array([[4, 7], 
              [2, 6]])

# 逆行列
A_inv = np.linalg.inv(A)
print("逆行列:\n", A_inv)

# 相似変換
P = np.array([[1, 0], 
              [0, 1]])  # 任意の変換行列
B = np.dot(np.linalg.inv(P), np.dot(A, P))
print("相似変換後の行列:\n", B)
```

---

## 随伴行列

### 随伴行列とは
行列 \( A \) の随伴行列（またはエルミート共役） \( A^* \) は、行列 \( A \) の転置行列 \( A^T \) に対してすべての要素を複素共役にしたものです。

#### 数式
\[
A^* = \overline{A^T}
\]

---

### Pythonで随伴行列を計算する
#### Pythonコード例
```python
import numpy as np

# 複素数行列
A = np.array([[1+2j, 2-1j], 
              [3+4j, 4-3j]])

# 随伴行列
A_adjoint = np.conjugate(A.T)
print("随伴行列:\n", A_adjoint)
```

---

## 行列計算の手間を測る

### 行列計算の効率性
行列計算の効率性は、行列のサイズとアルゴリズムに依存します。大規模な行列では計算コストが高くなるため、効率的なアルゴリズムや専用ライブラリの利用が必要です。

---

### Pythonで計算時間を測定する
Pythonの`time`モジュールを使って行列計算の時間を測定します。

#### Pythonコード例
```python
import numpy as np
import time

# ランダム行列
A = np.random.rand(1000, 1000)
B = np.random.rand(1000, 1000)

# 計算時間の測定
start = time.time()
C = np.dot(A, B)
end = time.time()

print("行列積の計算時間:", end - start, "秒")
```

# 第5章 行列の基本変形と不変量

この章では、行列の基本操作や性質に関する重要な概念を学びます。行列の階数、行列式、トレース、連立方程式、逆行列について、それぞれの意味を具体例を交えて解説します。Pythonを用いた実践例を通じて、実際の計算方法も学びます。

---

## 52 行列の階数

### 行列の階数とは
行列の階数（ランク）は、行列が表現する「独立した情報の量」を示す指標です。具体的には、行列の行や列の中で互いに独立しているものの最大数を表します。

#### 階数の役割
階数を調べることで、次のようなことがわかります：
- 行列がどれだけの次元を持つ線形空間を表現しているか。
- 行列のデータがどの程度冗長性を持っているか。

#### 例
行列
\[
A = \begin{bmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \\ 3 & 6 & 9 \end{bmatrix}
\]
は、すべての行が互いに線形従属しているため、階数は1です。

---

### Pythonで行列の階数を計算する
NumPyを使用して、行列の階数を簡単に計算できます。

#### Pythonコード例
```python
import numpy as np

# 行列
A = np.array([[1, 2, 3], 
              [2, 4, 6], 
              [3, 6, 9]])

# 階数を計算
rank = np.linalg.matrix_rank(A)
print("行列の階数:", rank)
```

---

## 53 行列式

### 行列式とは
行列式は、行列に対応する1つの値（スカラー）で、行列がどの程度空間を拡大または縮小するかを示します。

#### 行列式の意味
- **行列式が0の場合**: 行列は正則でなく、逆行列が存在しません。
- **行列式が0以外の場合**: 行列は正則で、逆行列が存在します。

#### 例: 2次の行列式
行列
\[
A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}
\]
の行列式は次のように計算されます：
\[
\det(A) = ad - bc
\]

---

### Pythonで行列式を計算する
#### Pythonコード例
```python
import numpy as np

# 行列
A = np.array([[2, 3], 
              [1, 4]])

# 行列式を計算
det = np.linalg.det(A)
print("行列式:", det)
```

---

## 54 トレース

### トレースとは
トレースは、行列の対角線上に並ぶ数字の合計です。たとえば：
\[
A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
\]
この行列のトレースは \( 1 + 4 = 5 \) です。

#### トレースの性質
- 行列の固有値（行列が空間をどれだけ伸縮させるかを示す値）の総和に等しい。
- 基底を変換しても（相似変換しても）トレースの値は変わりません。

---

### Pythonでトレースを計算する
#### Pythonコード例
```python
import numpy as np

# 行列
A = np.array([[1, 2], 
              [3, 4]])

# トレースを計算
trace = np.trace(A)
print("トレース:", trace)
```

---

## 55 連立方程式

### 連立方程式とは
連立方程式とは、複数の方程式を同時に満たす変数を求める問題です。中学で習う2本の方程式を解く手法に似ていますが、行列を使うことでより複雑な方程式を簡単に扱えます。

#### 行列を使った表現
次の連立方程式
\[
2x + y = 8 \\
x + 3y = 18
\]
は行列の形に書き直すと次のようになります：
\[
\begin{bmatrix} 2 & 1 \\ 1 & 3 \end{bmatrix} \cdot \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 8 \\ 18 \end{bmatrix}
\]

---

### Pythonで連立方程式を解く
NumPyを使って解を求めます。

#### Pythonコード例
```python
import numpy as np

# 係数行列
A = np.array([[2, 1], 
              [1, 3]])

# 定数ベクトル
b = np.array([8, 18])

# 方程式を解く
x = np.linalg.solve(A, b)
print("解:", x)
```

---

## 逆行列

### 逆行列とは
行列 \( A \) の逆行列 \( A^{-1} \) は、行列 \( A \) を用いた方程式を解くための道具で、次の性質を満たします：
\[
A \cdot A^{-1} = A^{-1} \cdot A = I
\]
ここで \( I \) は単位行列です。

#### 逆行列の性質
- 逆行列が存在する条件は、行列式が0でないことです。
- 行列が正則である場合、逆行列が存在します。

---

### Pythonで逆行列を計算する
#### Pythonコード例
```python
import numpy as np

# 行列
A = np.array([[2, 1], 
              [1, 3]])

# 逆行列を計算
A_inv = np.linalg.inv(A)
print("逆行列:\n", A_inv)
```

# 第6章 内積とフーリエ展開

この章では、内積、正規直交系、フーリエ展開、関数空間などの概念を学びます。フーリエ展開は、複雑な波形をシンプルな三角関数の組み合わせで表現する強力な数学ツールで、音声解析や信号処理で広く使われています。それぞれの内容をわかりやすく、中学生でも理解できるイメージで解説します。

---

## 62 正規直交系とフーリエ展開

### 正規直交系とは
正規直交系とは、数学でいう「互いに直角で、サイズが揃った基準セット」のことです。このセットを使うと、空間の中の任意の点（またはベクトル）を表現できます。

#### 条件
1. **直交（Orthogonal）**: 互いに直角になっている。
2. **正規（Normalized）**: 長さが1で揃っている。

#### イメージ
2次元平面では、\((1, 0)\) と \((0, 1)\) が正規直交系です。これを使えば、任意の点を簡単に座標で表現できます。

---

### フーリエ展開とは
フーリエ展開は、複雑な波を単純な三角関数（サイン波とコサイン波）の組み合わせで表現する方法です。

#### イメージ
フーリエ展開は「複雑な音をいくつかの単純な音（音階）に分解する」ような作業です。たとえば、音楽の中でピアノやギターの音を個別に取り出すようなものです。

#### 数式での表現
フーリエ展開を使うと、関数 \( f(x) \) は次のように表せます：
\[
f(x) = a_0 + \sum_{n=1}^\infty \left( a_n \cos(nx) + b_n \sin(nx) \right)
\]
ここで \( a_n \) と \( b_n \) はフーリエ係数と呼ばれる値で、関数をどのように分解するかを決定します。

---

### Pythonでフーリエ展開を計算する
以下はフーリエ展開の基本的な例です。

#### Pythonコード例
```python
import numpy as np
import matplotlib.pyplot as plt

# サンプル関数: サイン波
x = np.linspace(0, 2 * np.pi, 1000)
f = np.sin(x)

# フーリエ変換
fft_coefficients = np.fft.fft(f)
frequencies = np.fft.fftfreq(len(x), d=(x[1] - x[0]))

# 結果をプロット
plt.plot(frequencies, np.abs(fft_coefficients))
plt.title("フーリエ係数の大きさ")
plt.xlabel("周波数")
plt.ylabel("振幅")
plt.show()
```

---

## 63 関数空間

### 関数空間とは
関数空間は、関数そのものを「ベクトル」のように扱う数学的な空間です。たとえば、すべての2次関数の集合を1つの空間として見ることができます。

#### イメージ
関数空間は、普通の座標空間（点や線の集まり）を関数で置き換えたものと考えるとわかりやすいです。「関数の世界」をイメージしてください。

---

## 64 最小2乗法三角級数フーリエ級数

### 最小2乗法とは
最小2乗法は、データに最も近い曲線を見つける方法です。誤差を最小にするように計算するため、データの予測や近似に使われます。

#### イメージ
実験データに「ぴったりフィットする曲線」を探す方法です。グラフに点がばらついている場合でも、平均的な傾向を表す線を引くことができます。

---

## 65 直交関数系

### 直交関数系とは
直交関数系は、正規直交系の「関数バージョン」です。複数の関数が互いに直交している（独立している）集合を指します。

#### 例
\(\sin(nx)\) と \(\cos(nx)\) は、直交関数系を構成します。

---

## 66 ベクトル列の収束

### 収束とは
収束とは、繰り返し計算することで、値が特定の点に近づいていく現象を指します。ベクトル列の場合、繰り返し計算で得られるベクトルが、最終的に1つのベクトルに近づくかを調べます。

#### イメージ
たとえば、次の計算を繰り返すと、値はだんだん \( 2 \) に近づきます：
\[
x_1 = 1,\quad x_2 = 1.5,\quad x_3 = 1.75,\quad \dots
\]

---

## 67 フーリエ解析

### フーリエ解析とは
フーリエ解析は、フーリエ展開を使って複雑な波形を解析する手法です。音声データや信号データを細かく分解して、それぞれの成分を調べることができます。

#### イメージ
「複雑な音を分解して、それぞれの楽器がどんな音を出しているかを調べる方法」です。

---

### Pythonでフーリエ解析を実行する
#### Pythonコード例
```python
import numpy as np
import matplotlib.pyplot as plt

# サンプルデータ: 複雑な波
x = np.linspace(0, 2 * np.pi, 1000)
f = np.sin(x) + 0.5 * np.sin(3 * x)

# フーリエ変換
fft_coefficients = np.fft.fft(f)
frequencies = np.fft.fftfreq(len(x), d=(x[1] - x[0]))

# 結果をプロット
plt.plot(frequencies[:len(x)//2], np.abs(fft_coefficients)[:len(x)//2])
plt.title("フーリエ解析")
plt.xlabel("周波数")
plt.ylabel("振幅")
plt.show()
```


# 第7章 固有値と固有ベクトル

この章では、固有値と固有ベクトルについて学びます。これらは、行列が空間にどのように作用するかを調べる重要なツールです。さらに、行列の対角化やノルム、行列の関数といった応用的な内容も扱います。それぞれを中学生でもわかりやすいイメージで解説し、Pythonを用いた具体例を示します。

---

## 固有値

### 固有値とは
固有値は、行列が特定のベクトルを「方向を変えずに伸縮」させるときの伸縮の度合いを示す値です。

#### イメージ
- 固有値は「行列がベクトルを引っ張ったり縮めたりするときの倍率」です。
- ベクトルの向きは変わらず、大きさだけが変わるときに固有値が現れます。

#### 数式での定義
行列 \( A \) の固有値 \( \lambda \) と固有ベクトル \( v \) は次の関係を満たします：
\[
A \cdot v = \lambda \cdot v
\]
ここで、
- \( v \): 固有ベクトル（零ベクトル以外）
- \( \lambda \): 固有値

---

### Pythonで固有値を計算する
NumPyを使用して行列の固有値を簡単に計算できます。

#### Pythonコード例
```python
import numpy as np

# 行列
A = np.array([[4, 2], 
              [1, 3]])

# 固有値と固有ベクトルを計算
eigenvalues, eigenvectors = np.linalg.eig(A)

print("固有値:", eigenvalues)
print("固有ベクトル:\n", eigenvectors)
```

---

## 対角化

### 対角化とは
対角化は、行列を対角行列（対角線上に数値が並び、それ以外が0の行列）に変換する操作です。

#### メリット
- 行列の性質が簡単に理解できる。
- 行列の累乗や指数関数の計算が容易になる。

#### 数式
行列 \( A \) を対角化すると次のように表せます：
\[
A = P \cdot D \cdot P^{-1}
\]
ここで、
- \( D \): 対角行列（固有値が対角成分として並んでいる）。
- \( P \): 固有ベクトルを列として並べた行列。

---

### Pythonで対角化を実行する
以下は対角化の例です。

#### Pythonコード例
```python
import numpy as np

# 行列
A = np.array([[4, 2], 
              [1, 3]])

# 固有値と固有ベクトルを計算
eigenvalues, eigenvectors = np.linalg.eig(A)

# 対角行列
D = np.diag(eigenvalues)

# 固有ベクトル行列
P = eigenvectors

# 確認: A = P * D * P^-1
A_reconstructed = P @ D @ np.linalg.inv(P)
print("元の行列:\n", A)
print("再構成された行列:\n", A_reconstructed)
```

---

## 行列ノルムと行列の関数

### 行列ノルムとは
行列ノルムは、行列の「大きさ」を測るための基準です。これは、ベクトルの長さを測る定規のようなものと考えるとわかりやすいです。

#### フロベニウスノルム
フロベニウスノルムは、行列の要素をすべて2乗し、その総和の平方根をとった値です：
\[
\|A\|_F = \sqrt{\sum_{i,j} a_{ij}^2}
\]

---

### 行列の関数とは
行列の関数は、行列に対して数学的な関数（例えば指数関数や対数関数）を適用する操作です。

#### イメージ
行列の関数は、通常の関数を行列に拡張したものです。たとえば、\(\exp(x)\) を行列に適用すると、行列の指数関数を得ることができます。

---

### Pythonで行列ノルムと行列の関数を計算する
以下にPythonを用いた例を示します。

#### Pythonコード例
```python
import numpy as np
from scipy.linalg import expm

# 行列
A = np.array([[4, 2], 
              [1, 3]])

# フロベニウスノルムを計算
frobenius_norm = np.linalg.norm(A, 'fro')
print("フロベニウスノルム:", frobenius_norm)

# 行列の指数関数
A_exp = expm(A)
print("行列の指数関数:\n", A_exp)
```


# 第8章 ジョルダン標準形とスペクトル集合

この章では、ジョルダン標準形、スペクトル集合、行列の冪乗、ペロンフロベニウスの定理について学びます。これらの概念は高度に感じられるかもしれませんが、それぞれを具体例とともにわかりやすく解説し、Pythonを使った計算例を示します。

---

## ジョルダン標準形

### ジョルダン標準形とは
ジョルダン標準形は、行列を「できるだけ簡単な形」に変換する方法です。特定の基底を選ぶことで、任意の行列をジョルダン標準形に変換できます。

#### イメージ
ジョルダン標準形は、行列を「整った形」に整頓するような操作です。数式を因数分解して見やすくするのに似ています。

#### メリット
ジョルダン標準形に変換すると、行列の性質を簡単に理解したり、計算が効率的に行えたりします。

---

### Pythonでジョルダン標準形を計算する
#### Pythonコード例
```python
import numpy as np
from scipy.linalg import schur

# 行列
A = np.array([[4, 1], 
              [0, 4]])

# ジョルダン標準形の近似: Schur分解を使用
T, Z = schur(A)
print("ジョルダン標準形に近い行列:\n", T)
```

---

## ジョルダン分解と行列の冪乗

### ジョルダン分解とは
ジョルダン分解では、行列 \( A \) を次の形に分解します：
\[
A = P \cdot J \cdot P^{-1}
\]
ここで、
- \( J \): ジョルダン標準形
- \( P \): 固有ベクトルを基底とした行列

---

### 行列の冪乗
ジョルダン標準形を使うと、行列の累乗 \( A^n \) を効率的に計算できます：
\[
A^n = P \cdot J^n \cdot P^{-1}
\]

#### Pythonコード例
```python
import numpy as np

# 行列
A = np.array([[4, 1], 
              [0, 4]])

# 行列の累乗
n = 3
A_power = np.linalg.matrix_power(A, n)
print(f"行列の{n}乗:\n", A_power)
```

---

## 行列のスペクトル集合

### スペクトル集合とは
スペクトル集合は、行列の固有値を集めたものです。行列の本質的な性質を反映し、行列の操作や変形において重要な役割を果たします。

#### イメージ
スペクトル集合は、行列がどんな「倍率」でベクトルを伸ばしたり縮めたりするかを表す「リスト」のようなものです。

---

### Pythonでスペクトル集合を計算する
#### Pythonコード例
```python
import numpy as np

# 行列
A = np.array([[4, 2], 
              [1, 3]])

# 固有値を計算
eigenvalues = np.linalg.eigvals(A)
print("スペクトル集合:", eigenvalues)
```

---

## ペロンフロベニウスの定理

### ペロンフロベニウスの定理とは
ペロンフロベニウスの定理は、非負行列（すべての要素が0以上の行列）の固有値に関する性質を示した定理です。

#### 定理の内容
1. 最大の固有値（ペロン固有値）は実数で、0より大きい。
2. 対応する固有ベクトルのすべての成分は正（または非負）です。

#### イメージ
行列が持つ「最大の倍率」と、その倍率に対応するベクトルの形状を調べる方法です。

---

### Pythonでペロンフロベニウスの定理を確認する
#### Pythonコード例
```python
import numpy as np

# 非負行列
A = np.array([[2, 1], 
              [1, 3]])

# 最大固有値を計算
eigenvalues, eigenvectors = np.linalg.eig(A)
max_eigenvalue = max(eigenvalues)
max_eigenvector = eigenvectors[:, np.argmax(eigenvalues)]

print("最大固有値:", max_eigenvalue)
print("対応する固有ベクトル:\n", max_eigenvector)
```


# 第9章 力学系

この章では、力学系の基本的な概念を学びます。ニュートンの運動方程式や線形の微分方程式、確率的なシステムを記述するマルコフ過程やランダムフィールド、さらに1径数半群と生成行列に至るまで、幅広いテーマを扱います。それぞれの概念を初心者にもわかりやすく、中学生でも理解しやすい形で解説し、Pythonを使った具体例を示します。

---

## ニュートンの運動方程式

### ニュートンの運動方程式とは
ニュートンの運動方程式は、物体の運動を記述する基本的な方程式で、次の形で表されます：
\[
F = m \cdot a
\]
ここで、
- \( F \): 力（ニュートン）
- \( m \): 質量（キログラム）
- \( a \): 加速度（メートル毎秒毎秒）

#### イメージ
ニュートンの運動方程式は「物を動かすために必要な力」を計算する公式です。例えば、重い物を押すには軽い物を押すよりも大きな力が必要です。

---

### Pythonで運動方程式をシミュレーション
物体が動く様子をPythonでシミュレーションします。

#### Pythonコード例
```python
import numpy as np
import matplotlib.pyplot as plt

# 初期設定
mass = 2.0  # 質量 (kg)
force = 10.0  # 力 (N)
time = np.linspace(0, 5, 100)  # 時間 (秒)

# 加速度と速度
acceleration = force / mass
velocity = acceleration * time

# プロット
plt.plot(time, velocity)
plt.title("速度の時間変化")
plt.xlabel("時間 (秒)")
plt.ylabel("速度 (m/s)")
plt.show()
```

---

## 線形の微分方程式

### 微分方程式とは
微分方程式は、関数とその微分の関係を表す方程式です。線形の微分方程式は次の形を持ちます：
\[
\frac{dy}{dt} + p(t)y = q(t)
\]
ここで、\( p(t) \) と \( q(t) \) は与えられた関数です。

#### イメージ
微分方程式は「変化を記述する式」です。たとえば、水の温度が時間とともにどう変化するかを記述するのに使われます。

---

### Pythonで微分方程式を解く
SciPyを使って微分方程式を解きます。

#### Pythonコード例
```python
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

# 微分方程式の定義: dy/dt = -2y + 1
def diff_eq(t, y):
    return -2 * y + 1

# 初期条件と時間範囲
y0 = 0  # 初期値
t_span = (0, 5)  # 時間の範囲
t_eval = np.linspace(0, 5, 100)

# 解を求める
solution = solve_ivp(diff_eq, t_span, [y0], t_eval=t_eval)

# プロット
plt.plot(solution.t, solution.y[0])
plt.title("微分方程式の解")
plt.xlabel("時間 (t)")
plt.ylabel("y(t)")
plt.show()
```

---

## 定常マルコフ過程の平衡状態

### マルコフ過程とは
マルコフ過程は、未来の状態が現在の状態のみに依存する確率的なプロセスです。たとえば、天気予報で「明日の天気は今日の天気だけで決まる」という考え方がこれに該当します。

#### 定常状態
マルコフ過程の定常状態（平衡状態）とは、時間が経つと確率が変わらなくなる状態を指します。

---

### Pythonでマルコフ過程をシミュレーション
#### Pythonコード例
```python
import numpy as np

# 遷移行列
P = np.array([[0.9, 0.1],
              [0.2, 0.8]])

# 初期状態
state = np.array([1, 0])  # 最初は状態1

# 定常状態に近づく
for _ in range(10):
    state = np.dot(state, P)
    print("現在の状態:", state)
```

---

## マルコフランダムフィールド

### マルコフランダムフィールドとは
マルコフランダムフィールド（MRF）は、確率的なシステムをグラフで表現する手法です。特定のノード（点）が、隣接するノードにのみ依存するという性質を持っています。

#### イメージ
MRFは「ネットワーク上でどんな影響が隣同士に伝わるか」を調べる仕組みです。例えば、ソーシャルネットワークで友達関係の影響を分析するのに使います。

---

## 1径数半群と生成行列

### 1径数半群とは
1径数半群は、時間的に連続した変化を記述する数学的な構造で、行列指数関数を用いて次のように記述されます：
\[
P(t) = \exp(tG)
\]
ここで、\( G \) は生成行列と呼ばれる行列です。

---

### Pythonで行列指数関数を計算する
#### Pythonコード例
```python
from scipy.linalg import expm
import numpy as np

# 生成行列
G = np.array([[-1, 1], 
              [1, -1]])

# 時間 t における行列
t = 2.0
P_t = expm(t * G)
print("時刻 t =", t, "における行列:\n", P_t)
```

