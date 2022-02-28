# Forecast_of_industrial_steam_volume

ボイラーの状態から発生する蒸気量を予測する。

データは、トレーニングデータ(train.txt)とテストデータ(test.txt)に分けられ、フィールド「V0」-「V37」の38フィールドが特徴変数、ターゲット変数「target」となる。

テストデータの目的変数を予測し、標準はMSE (mean square error)。
