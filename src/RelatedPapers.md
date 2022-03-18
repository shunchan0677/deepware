## Related papers

* Deepな位置推定手法の調査
  * DeepWare作るためにDeepなLocalizationがほしいので調査
  * MappingはHDMapと一致しないといけないからなくても可能。

* 調べた論文
  * DeepLocalization(https://arxiv.org/pdf/1904.09007.pdf)
  * PoseMap: Lifelong, Multi-Environment 3D LiDAR Localization(https://research.csiro.au/robotics/wp-content/uploads/sites/96/2018/09/posemap_iros2018-10.pdf)
  * LO-Net:(https://ssk0109.hatenablog.com/entry/2019/04/21/081207)
  * Uber Localization(http://proceedings.mlr.press/v87/barsan18a/barsan18a.pdf)

* Lo-Net
  * LOはLidar Odometry。学習ベースな手法でモデルベースの手法のSOTAであるLOAMと同等の精度を達成
  * End-to-EndでOdometry estimation(Localization)とMappingが可能
  * mask-weighted geometric constraint lossによって効率的に特徴量抽出を学習
  ＊ 手法では点群を2つ入力し、オドメトリを回帰する。ネットワークはNormal estimation sub-network, a mask prediction sub-network, and a
Siamese pose regression main-networkの3つを同時に学習する。マッピングは別で行う。
  * 逐次学習を行う。
  * Lidar点群は3次元ではなく, 2次元のrenge imageっぽく変換してCNNに入力。
  * Normal estimationでは各点に対して大きさ1の法線を推定する。ここでは、最近傍点に向かうベクトルにレンジ値から計算される重みをかけたものと推定値の内積を最小化することで法線を求める。これはPCAと等しい。
  * Pose Estimationでは位置とクォータニオンをl2で推定する。学習時に学習できる重みを変数としておいて2つのl2のスケールを調節する。
  * Mask EstimationではPose Estimationと共有のEncoderを持つEncoder-Decorderを学習し、Dynamic Objectを抽出できるような不確実性を推論するマップを出力する。Decorderにはskip connectionとfireDeconvを利用している。
  * MapとのMatchingについては書かれていない...?

* Deep Localization
  * こちらは位置推定の話。現在の点群とマップ点群をPointNetに入れて移動分デルタを推定。
  * 実際にはEKFで予測値をフィルタリングして結果を抽出する。
  * 手法は理解しやすいが, 結果の比較が貧弱なのでなんとも言えない...。
  

* FutureSLAM構想の具体化
  * SLAP -> Simultaneous Localization and Prediction
    * P-SLAM(prediction based Slam)っていうのがあるらしい。(https://www.semanticscholar.org/paper/P-SLAM%3A-Simultaneous-Localization-and-Mapping-With-Chang-Lee/67832b02326336c98e81495d1bb85947fd9ee6d2)
  * 次ステップの点群生成とそれを使った自己位置推定を交互に行う
  * GAN, Cycle-lossを導入して教師なしにもできそう
  * OGM-Pred部分を点群推定に代替すれば実現できそう？
  * そもそも次フレーム点群推定のほうが将来位置推定より難しくない？？

* PRECOG - ICML(https://arxiv.org/pdf/1905.01296.pdf)
  * 高さで2つに分けた点群を入れて自己車両も含めた周辺車両を物体認識×Prediction。
  * 中間補完を使ってスムージング
  
* Learning by Cheating -  http://vladlen.info/papers/learning-by-cheating.pdf
  * M2M学習してからE2Eに蒸留する？
