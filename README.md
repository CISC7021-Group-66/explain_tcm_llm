# Welcome ❤️‍🔥 !

這裡是 2024 `CISC7021 Applied Natural Language Processing` 課程 **Group 66** 的 Final Project ! ❤️‍🔥💥💫🌟✨

這個倉庫主要包含所有的測試代碼，方便版本回滾，能多 commit 就多 commit 吧！

# Useful Command

## 版本管理
如果遠端main與本地main不同，希望遠端覆蓋本地（忽略所有本地更改），`git reset --hard origin/main`

## 查看顯存使用
終端使用：
```Terminal
watch -n 1 nvidia-smi
```
每秒刷新顯存使用。

### 強行終止python進程以釋放CUDA、顯存
終端使用：
```Terminal
sudo pkill -9 python
```

## 創建版本要求
對於 `ipynb` 需使用 `pip install pipreqsnb` 。

在項目目錄中打開終端運行(例子) `pipreqsnb ./lime-zhongjing-jieba.ipynb`。

會生成一個 `requirements.txt` ，包含 `lime-zhongjing-jieba.ipynb` 所包含的所有 `import` 模組的版本，用於寫報告。
就像這樣：
```
jieba==0.42.1
lime==0.2.0.1
matplotlib==3.9.2
numpy==2.1.3
pandas==2.2.3
torch==2.5.1
transformers==4.46.2
```


## 安裝版本要求
在終端使用命令：
```
pip install -r requirements.txt
```
會安裝 `requirements.txt` 中制定的模組和制定的版本，確保與實驗環境一致。
