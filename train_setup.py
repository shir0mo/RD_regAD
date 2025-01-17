import os
import shutil
import pandas as pd

# データセットのパスを設定
mvtec_ad_path = '../mpdd'  # MVTec ADデータセットのパス
output_path = '../mpdd_train'  # 新しいデータセットの保存先

def make_train_dataset(categories, mvtec_ad_path, output_path):
    # ラベル情報を保存するためのリスト
    data = []

    # 各カテゴリに対して処理を行う
    for category in categories:
        print(f"Processing category: {category}")
        
        # 正常画像のディレクトリパス
        normal_image_dir = os.path.join(mvtec_ad_path, category, 'train', 'good')
        
        # 出力カテゴリディレクトリを作成
        output_category_dir = os.path.join(output_path, category)
        if not os.path.exists(output_category_dir):
            os.makedirs(output_category_dir)
        
        # 正常画像を出力ディレクトリにコピー
        for img_name in os.listdir(normal_image_dir):
            img_path = os.path.join(normal_image_dir, img_name)
            output_img_path = os.path.join(output_category_dir, img_name)
            
            # 画像をコピー
            shutil.copy(img_path, output_img_path)
            
            # ラベル情報をリストに追加
            data.append([output_img_path, category])

    # ラベル情報をCSVファイルに保存
    df = pd.DataFrame(data, columns=['file_path', 'label'])
    df.to_csv(os.path.join(output_path, 'labels.csv'), index=False)

    print("Dataset creation completed.")

# カテゴリ一覧を取得
categories = [d for d in os.listdir(mvtec_ad_path) if os.path.isdir(os.path.join(mvtec_ad_path, d))]

# 出力ディレクトリが存在しない場合は作成
if not os.path.exists(output_path):
    print("Train dataset is not available.")
    os.makedirs(output_path)
    make_train_dataset(categories, mvtec_ad_path, output_path)
else:
    print("Train dataset is available.")

