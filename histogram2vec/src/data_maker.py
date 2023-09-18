# -*- coding: utf-8 -*-
"""
Created on Fri 29 15:46:32 2022

prepare dataloader

@author: tadahaya
"""
import time
import datetime
from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from itertools import chain

FIELD = [
    "Dataset",	"DiseaseType", "SpecimenID", "Enzyme", "Position", "ID",
    "Number", "X", "Y", "FITC", "mCherry", "Date"
    ]

class Preprocess:
    """
    preprocessor, 基本的にハード
    
    ID: 2nd_H_0404_6PGDH2_1
        2ndデータセット
            1st, 2nd, astと三つある. astってなんだろうか
        健常人ラベル
            healty OR cancerで問題ない
        血液検体IDが0404
            検体IDで問題ない
        6PGDH2
            これかLDHの二つなので見ている酵素だろう
        1
            1-12までしかないので視野番号だろう
            Positionと他では表記されていた
    Number
        各視野内での輝点番号
    X, Y
        各視野内でのX, Y座標
    FITC
        プローブ蛍光由来の強度
    mCherry
        reference色素の強度
    Date
        測定日付
    
    """
    def __init__(
            self, url, fileout:str="", sep:str=",", v_name:str="FITC"
            ):
        df = pd.read_csv(url, sep=sep)
        ids = df["ID"].values.flatten().tolist()
        split = []
        n_item = 0
        for i in ids:
            tmp = i.split("_")
            if n_item==0:
                n_item = len(tmp)
            else:
                if n_item!=len(tmp):
                    print(i)
                    raise ValueError
            split.append(tmp)
        df_id = pd.DataFrame(split)
        self.id_col = ["Dataset", "DiseaseType", "SpecimenID", "Enzyme", "Position"]
        df_id.columns = self.id_col
        comb = pd.concat([df_id, df], axis=1, join="inner")
        print(df_id.shape, df.shape)
        self.data = comb
        # preprocessing
        self._fix(v_name)
        # export
        if len(fileout) == 0:
            ext = url.split(".")[-1]
            fileout = url.replace(ext, f"_modified.{ext}")
        self.data.to_csv(fileout, sep=sep)


    def _fix(self, v_name:str="FITC"):
        """ mainの値であるFITCに変なstrが混じっていたので対処 """
        before = self.data.shape
        def convert(x):
            try:
                return float(x)
            except ValueError:
                return np.nan
        self.data.loc[:, v_name] = self.data.loc[:, v_name].map(convert)
        self.data = self.data.dropna(subset=[v_name])
        print(before, self.data.shape)


    def check_content(self):
        """ 中身を確認する """
        count = []
        for c in self.id_col:
            print("====================================", c)
            tmp = self.data[c].values.flatten().tolist()
            cntr = Counter(tmp)
            count.append(cntr)
            for k, v in cntr.items():
                print(k, v)


class Data:
    """
    data格納モジュール, 基本的にハード
        
    """
    def __init__(self, url:str=None, df:pd.DataFrame=None):
        # 読み込み
        self.data = None
        if url is None:
            if df is None:
                raise ValueError("!! Provide url or dataframe !!")
            else:
                self.data = df
        else:
            self.data = pd.read_csv(url, index_col=0)
        # データの中身の把握
        col = list(self.data.columns)
        self.dic_components = dict()
        for c in col:
            tmp = df[c].values.flatten().tolist()
            self.dic_components[c] = Counter(tmp)


    def conditioned(self, condition:dict):
        """ 解析対象とするデータを条件付けする """
        for k, v in condition.items():
            try:
                self.data = self.data[self.data[k]==v]
            except KeyError:
                raise KeyError("!! Wrong key in condition: check the keys of condition !!")


    def sample(
        self, sid:int, n_sample:int=256, ratio:float=0.9,
        v_name:str="FITC", s_name:str="SpecimenID"
        ):
        """
        指定した検体からn_sampleの回の輝点のサンプリングを行う

        Parameters
        ----------
        sid: int
            Specimen ID, 検体をidentifyする

        """
        tmp = self.data[self.data[s_name]==sid]
        dim = int(tmp.shape[0] * ratio)
        res = np.zeros((n_sample, dim))
        for i in range(n_sample):
            res[i, :] = tmp.sample(n=dim)[v_name].values
        return res


    def prep_data(
            self, samplesize:int=10000, ratio:float=0.9,
            shuffle:bool=True, v_name:str="FITC", s_name:str="SpecimenID"
            ):
        """ 指定したsamplesizeまでサンプリングを行う """
        specimens = set(list(self.data[s_name]))
        n_sample = samplesize // len(specimens) # n_sampleを決める
        res = []
        for s in specimens:
            tmp = self.sample(s, n_sample, ratio, v_name, s_name)
            tmp = [v[0] for v in np.split(tmp, n_sample, axis=0)]
            ## 1個中に入るため
            res.append(tmp)
        res = list(chain.from_iterable(res))
        if shuffle:
            rng = np.random.default_rng()
            rng.shuffle(res)
        return res


class DataMaker:
    """
    dataを読み込んでヒストグラムを表すnp配列へと変換する
    
    """
    def __init__(self, pixel:tuple=(256, 256)):
        self.pixel = pixel
        self._dpi = 100
        self._figsize = pixel[0] / self._dpi, pixel[1] / self._dpi
        self.data = None


    def set_data(self, data):
        """ setter """
        self.data = data


    def data2histo(self, fileout:str="", ratio:float=0.9, bins=(30, 25)):
        """
        dataをまとめてhistogram arrayへと変換, npzで保存する
        input array, output arrayの順
        各arrayはsample, h, wの順
        
        Parameters
        ----------
        ratio: float, (0, 1)
            サンプルからデータ点を取得する割合
        
        bins: tuple
            前者がinput, 後者がoutputのヒストグラムのbinsを指定する
            AE的に組むため, 前者が大きいことが前提

        """
        assert len(fileout) > 0
        assert (ratio > 0) & (ratio < 1) 
        assert bins[0] >= bins[1]
        # dataの準備
        array0 = np.zeros((len(self.data), self.pixel[0], self.pixel[1]))
        array1 = np.zeros((len(self.data), self.pixel[0], self.pixel[1]))
        for d in trange(len(self.data)):
            # dataのhistogram化
            img0 = self.get_hist_array(d, bins=bins[0])
            img1 = self.get_hist_array(d, bins=bins[1])
            # imageのbinarize化と格納
            array0[i, :, :] = self.binarize(img0)
            array1[i, :, :] = self.binarize(img1)
        # npzで保存
        np.savez_compressed(fileout, array0, array1)


    def get_hist_array(self, data, bins=30):
        """
        データをヒストグラムへと変換し, そのarrayを得る
        
        """
        assert bins[0] >= bins[1]
        # prepare histogram
        fig = plt.figure(figsize=self._figsize, dpi=self._dpi)
        ax = fig.add_subplot(1, 1, 1)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["bottom"].set_visible(False)
        plt.gca().spines["left"].set_visible(False)
        plt.tick_params(
            labeltop=False, labelright=False, labelbottom=False, labelleft=False,
            top=False, right=False, bottom=False, left=False
            )
        ax.hist(data, color="black", bins=bins)
        # convert array
        fig.canvas.draw() # レンダリング
        data = fig.canvas.tostring_rgb() # rgbのstringとなっている
        w, h = fig.canvas.get_width_height()
        c = len(data) // (w * h) # channelを算出
        img = np.frombuffer(data, dtype=np.uint8).reshape(h, w, c)
        plt.close()
        return img


    def binarize(self, data):
        """ 得られたarrayを二値化する """
        data = (data == 0).sum(axis=2) # h, w, cであり, blackなので255, 0のみとなっている
        data = (data > 0).astype(np.uint8)
        return data