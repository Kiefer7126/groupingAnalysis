## スペクトログラム
* ｙ軸は対数
* STFT配列は[f, t]
    - frequency bin f at frame t

#### 1サンプル1ピクセルで描画したい
* 画像の幅 = stft配列のtのサンプル数
* 画像の高さ = stft配列のfのサンプル数（対数だとダメ？）
* スペクトログラムの横の1ピクセルは（siftSize / sr）

## GLCM
#### scikit-imageの仕様
* glcm = greycomatrix(gray256Image, [距離], [方向], levels=256, normed=True, symmetric=True)
* 方向は0, 45, 90, 135を使用
* glcm[d][a] d番目の距離とa番目の方向

#### glcm構造体
* 方向毎に作成
* contrast = [bin1, bin2, bin3, ... ,]
* 距離が小さいとあまり45度と135度に差はないが距離を大きくすると差が大きくなる

## 階層的クラスタリング
#### method
* single(単結合法，最短距離法)：空間濃縮
    - 併合されてできたクラスタは以後併合され__やすい__
* complete(完全連結法，最長距離法)：空間拡散
    - 併合されてできたクラスタは以後併合され__にくい__
* ward(ウォード法)：空間拡散
* average(群平均法)：空間保存
* centroid：空間保存
* weighted：空間保存
* median：空間保存

## 関数の説明
#### calcGLCM
* 画像からGLCMを計算する
* 引数：画像のpath  
* 返り値：GLCM

#### calcGLCMfeatures
* GLCMの各統計的特徴量を計算する
* 引数：特徴量構造体(glcmFeatures)

#### calcGLCMfeaturesDistance
* GLCMの各統計的特徴量の隣接する距離を求める
* 引数：特徴量構造体(glcmFeatures)

#### calcDistance
* ベクトルのn番目とn+1番目の距離を求める
* 引数：特徴量ベクトル
* 返り値：距離ベクトル

#### divideSpectrogramByBeat
* 拍でスペクトログラムを分割し，画像として保存する
* 引数：path，拍節構造，サンプリングレート
* 返り値：分割されたスペクトログラム（使用していない説）

#### divideSpectrogram
* スペクトログラムを指定した定窓幅で分割し，画像として保存する
* 引数：path
* 返り値：分割されたスペクトログラム（使用していない説）

#### drawSpectrogram
* スペクトログラムを描画し，画像として保存する

#### loadBeat
* 拍節構造を読み込む
