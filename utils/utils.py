# coding:utf-8

import ast
import math
import pickle
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, recall_score, confusion_matrix, precision_score
import os
import random
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AdamW
import wfdb
from prettytable import PrettyTable


class FGM(object):  # 这个类的作用是对抗训练
    """
    基于FGM算法的攻击机制

    Args:
        module (:obj:`torch.nn.Module`): 模型

    Examples::

        >>> # 初始化
        >>> fgm = FGM(module)
        >>> for batch_input, batch_label in data:
        >>>     # 正常训练
        >>>     loss = module(batch_input, batch_label)
        >>>     loss.backward() # 反向传播，得到正常的grad
        >>>     # 对抗训练
        >>>     fgm.attack() # 在embedding上添加对抗扰动
        >>>     loss_adv = module(batch_input, batch_label)
        >>>     loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        >>>     fgm.restore() # 恢复embedding参数
        >>>     # 梯度下降，更新参数
        >>>     optimizer.step()
        >>>     optimizer.zero_grad()

    Reference:
        [1]  https://zhuanlan.zhihu.com/p/91269728
    """

    def __init__(self, module):
        self.module = module
        self.backup = {}

    def attack(
        self,
        epsilon=1.,
        emb_name='word_embeddings'
    ):
        for name, param in self.module.named_parameters():
            if param.requires_grad and param.grad is not None and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(
        self,
        emb_name='word_embeddings'
    ):
        for name, param in self.module.named_parameters():
            if param.requires_grad and emb_name in name:
                if name in self.backup:
                    param.data = self.backup[name]
        self.backup = {}


class AWP:  # 这个类的作用是对抗训练
    def __init__(
        self,
        model,
        adv_param="weight",
        adv_lr=1,
        adv_eps=0.2,
        adv_step=1,
    ):
        self.model = model
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.adv_step = adv_step
        self.backup = {}
        self.backup_eps = {}

    def attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(
                            param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )
                # param.data.clamp_(*self.backup_eps[name])

    def save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def restore(self,):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}


class EMA(object):  # 这个类的作用是ema累积模型参数
    """
    Maintains (exponential) moving average of a set of parameters.
    使用ema累积模型参数

    Args:
        parameters (:obj:`list`): 需要训练的模型参数
        decay (:obj:`float`): 指数衰减率
        use_num_updates (:obj:`bool`, optional, defaults to True): Whether to use number of updates when computing averages

    Examples::

        >>> ema = EMA(module.parameters(), decay=0.995)
        >>> # Train for a few epochs
        >>> for _ in range(epochs):
        >>>     # 训练过程中，更新完参数后，同步update shadow weights
        >>>     optimizer.step()
        >>>     ema.update(module.parameters())
        >>> # eval前，进行ema的权重替换；eval之后，恢复原来模型的参数
        >>> ema.store(module.parameters())
        >>> ema.copy_to(module.parameters())
        >>> # evaluate
        >>> ema.restore(module.parameters())

    Reference:
        [1]  https://github.com/fadel/pytorch_ema
    """  # noqa: ignore flake8"

    def __init__(
        self,
        parameters,
        decay,
        use_num_updates=True
    ):
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach()
                              for p in parameters if p.requires_grad]
        self.collected_params = []

    def update(self, parameters):
        """
        Update currently maintained parameters.
        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the same set of
            parameters used to initialize this object.
        """
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) /
                        (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay * (s_param - param))

    def copy_to(self, parameters):
        """
        Copy current parameters into given collection of parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored moving averages.
        """
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone()
                                 for param in parameters
                                 if param.requires_grad]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            if param.requires_grad:
                param.data.copy_(c_param.data)


def seed_everything(seed):  # 这个函数的作用是设置随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def compute_kl_loss(p, q, pad_mask=None):   # 这个函数的作用是计算kl散度

    p_loss = F.kl_div(F.log_softmax(p, dim=-1),
                      F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1),
                      F.softmax(p, dim=-1), reduction='none')

    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss


def build_optimizer(
        model,
        args,
        eps=1e-6,
        correct_bias=True): # 这个函数的作用是构建优化器
    # model_param = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.weight"]

    # if args.model != 'bert':
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # elif args.model == 'bert':
    #     bert_param_optimizer = []
    #     classifier_param_optimizer = []

    #     for name, param in model_param:
    #         space = name.split('.')
    #         if 'encoder' in space[0] or 'embeddings' in space[0]:
    #             bert_param_optimizer.append((name, param))
    #         elif 'resnet' in space[0] or 'classifier' in space[0]:
    #             classifier_param_optimizer.append((name, param))

    #     optimizer_grouped_parameters = [
    #         {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
    #             "weight_decay": args.weight_decay, 'lr': args.bert_lr},
    #         {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
    #             "weight_decay": 0.0, 'lr': args.bert_lr},

    #         {"params": [p for n, p in classifier_param_optimizer if not any(nd in n for nd in no_decay)],
    #             "weight_decay": args.weight_decay, 'lr': args.lr},
    #         {"params": [p for n, p in classifier_param_optimizer if any(nd in n for nd in no_decay)],
    #             "weight_decay": 0.0, 'lr': args.lr}
        # ]

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.lr,
                      eps=eps,
                      correct_bias=correct_bias,
                      weight_decay=args.weight_decay)
    return optimizer


def evaluate(predictions, labels, eval_mode=False): # 这个函数的作用是计算测试集准确率

    def mean_f1(predictions, labels):
        """
        通过遍历阈值（从 0.1 到 0.9），将 predictions 中的每个元素与阈值进行比较，
        然后计算对应的 F1 分数，并将其添加到列表 f1 中。最后返回 f1 列表的平均值。
        """
        f1 = []
        for i in range(1, 10):
            th = i / 10
            predictions_binary = predictions >= th
            score = f1_score(labels, predictions_binary, average='macro')
            f1.append(score)
        return np.mean(f1)
    f1 = mean_f1(predictions, labels)   # 计算平均 F1 score

    def mean_specificity(predictions, labels):
        """
        通过遍历阈值（从 0.1 到 0.9），将 predictions 中的每个元素与阈值进行比较，
        然后计算对应的特异性，并将其添加到列表 specificities 中。最后返回 specificities 列表的平均值。
        """
        specificities = []
        for i in range(1, 10):
            th = i / 10
            predictions_binary = predictions >= th
            # 对每个标签单独计算特异性
            for j in range(labels.shape[1]):
                tn, fp, fn, tp = confusion_matrix(labels[:, j], predictions_binary[:, j]).ravel()
                specificity = tn / (fp + tn)
                specificities.append(specificity)
        return np.mean(specificities)
    specificity = mean_specificity(predictions, labels)   # 计算平均特异性 specificity

    predictions = predictions >= 0.5    # 将 predictions 转换为二元预测结果

    accuracy = accuracy_score(labels, predictions)  # 计算准确率 accuracy
    recall = recall_score(labels, predictions, average='macro')  # 计算召回率 recall
    precision = precision_score(labels, predictions, average='macro')    # 计算精确率 precision
    auc = roc_auc_score(labels, predictions)    # 计算模型准确性 auc

    # if eval_mode:
    #     # class-wise
    #     class_dict = {0: 'AMI', 1: 'ASMI', 2: 'ILMI', 3: 'IMI', 4: 'NORM'}
    #     class_acc = {}
    #     for i in range(predictions.shape[1]):
    #         acc = accuracy_score(labels[:, i], predictions[:, i])
    #         class_acc[class_dict[i]] = acc        

    return {'accuracy': accuracy, 'recall': recall, 'mean_specificity': specificity, 'precision': precision, 'mean_f1': f1, 'auc': auc}


class Logs: # 这个类的作用是将结果写入log文件
    def __init__(self, path) -> None:
        self.path = path
        with open(self.path, 'w', encoding='utf-8') as f:
            f.write('')

    def write(self, content):
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(content)


def load_dataset(path, sampling_rate, release=False):   # 这个函数的作用是读取数据集

    # load and convert annotation data
    Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))  # 将标签数据转化为字典格式

    # Load raw signal data
    X = load_raw_data_ptbxl(Y, sampling_rate, path)

    return X, Y # X是信号数据，Y是标签数据


def load_raw_data_ptbxl(df, sampling_rate, path):   # 这个函数的作用是读取信号数据
    if sampling_rate == 100:
        if os.path.exists(path + 'raw100.npy'):
            data = np.load(path+'raw100.npy', allow_pickle=True)    # 读取信号数据，allow_pickle=True的作用是允许读取pickle格式的数据
        else:
            data = [wfdb.rdsamp(path+f) for f in tqdm(df.filename_lr)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path+'raw100.npy', 'wb'), protocol=4)    # 保存信号数据，pickle.dump()函数的作用是将数据保存为pickle格式，因为pickle格式的数据读取速度比较快
    elif sampling_rate == 500:
        if os.path.exists(path + 'raw500.npy'):
            data = np.load(path+'raw500.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path+f) for f in tqdm(df.filename_hr)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path+'raw500.npy', 'wb'), protocol=4)
    return data


def compute_label_aggregations(df, folder, ctype):  # 这个函数的作用是将标签数据转化为one-hot编码，one-hot编码的作用是将标签数据转化为模型可以识别的格式

    df['scp_codes_len'] = df.scp_codes.apply(lambda x: len(x))  # 计算每个样本的标签数量

    aggregation_df = pd.read_csv(folder+'scp_statements.csv', index_col=0)  # 读取标签数据

    if ctype in ['diagnostic', 'subdiagnostic', 'superdiagnostic']: # 这个if语句的作用是将标签数据转化为one-hot编码

        def aggregate_all_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    tmp.append(key)
            return list(set(tmp))

        def aggregate_subdiagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    c = diag_agg_df.loc[key].diagnostic_subclass
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        def aggregate_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    c = diag_agg_df.loc[key].diagnostic_class
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        diag_agg_df = aggregation_df[aggregation_df.diagnostic == 1.0]
        if ctype == 'diagnostic':
            df['diagnostic'] = df.scp_codes.apply(aggregate_all_diagnostic)
            df['diagnostic_len'] = df.diagnostic.apply(lambda x: len(x))
        elif ctype == 'subdiagnostic':
            df['subdiagnostic'] = df.scp_codes.apply(aggregate_subdiagnostic)
            df['subdiagnostic_len'] = df.subdiagnostic.apply(lambda x: len(x))
        elif ctype == 'superdiagnostic':
            df['superdiagnostic'] = df.scp_codes.apply(aggregate_diagnostic)
            df['superdiagnostic_len'] = df.superdiagnostic.apply(
                lambda x: len(x))
    elif ctype == 'form':
        form_agg_df = aggregation_df[aggregation_df.form == 1.0]

        def aggregate_form(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in form_agg_df.index:
                    c = key
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        df['form'] = df.scp_codes.apply(aggregate_form)
        df['form_len'] = df.form.apply(lambda x: len(x))
    elif ctype == 'rhythm':
        rhythm_agg_df = aggregation_df[aggregation_df.rhythm == 1.0]

        def aggregate_rhythm(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in rhythm_agg_df.index:
                    c = key
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        df['rhythm'] = df.scp_codes.apply(aggregate_rhythm)
        df['rhythm_len'] = df.rhythm.apply(lambda x: len(x))
    elif ctype == 'all':
        df['all_scp'] = df.scp_codes.apply(lambda x: list(set(x.keys())))
    elif ctype == 'mi':
        mi_agg_df = aggregation_df[(aggregation_df.diagnostic_class == 'MI') | (
            aggregation_df.diagnostic_class == 'NORM')] # 获取标签数据中diagnostic_class为MI或NORM的数据

        def aggregate_mi(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in mi_agg_df.index:
                    c = key
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))   # set()函数的作用是去除重复的元素，list()函数的作用是将数据转化为列表格式
        df['mi'] = df.scp_codes.apply(aggregate_mi)

    return df


def select_data(XX, YY, ctype, min_samples, outputfolder):  # 这个函数的作用是选择数据，min_samples的作用是选择样本数量大于min_samples的数据
    # convert multilabel to multi-hot
    mlb = MultiLabelBinarizer()
    if ctype == 'diagnostic':
        X = XX[YY.diagnostic_len > 0]
        Y = YY[YY.diagnostic_len > 0]
        mlb.fit(Y.diagnostic.values)
        y = mlb.transform(Y.diagnostic.values)
    elif ctype == 'subdiagnostic':
        counts = pd.Series(np.concatenate(
            YY.subdiagnostic.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.subdiagnostic = YY.subdiagnostic.apply(
            lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['subdiagnostic_len'] = YY.subdiagnostic.apply(lambda x: len(x))
        X = XX[YY.subdiagnostic_len > 0]
        Y = YY[YY.subdiagnostic_len > 0]
        mlb.fit(Y.subdiagnostic.values)
        y = mlb.transform(Y.subdiagnostic.values)
    elif ctype == 'superdiagnostic':
        counts = pd.Series(np.concatenate(
            YY.superdiagnostic.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.superdiagnostic = YY.superdiagnostic.apply(
            lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['superdiagnostic_len'] = YY.superdiagnostic.apply(lambda x: len(x))
        X = XX[YY.superdiagnostic_len > 0]
        Y = YY[YY.superdiagnostic_len > 0]
        mlb.fit(Y.superdiagnostic.values)
        y = mlb.transform(Y.superdiagnostic.values)
    elif ctype == 'form':
        # filter
        counts = pd.Series(np.concatenate(YY.form.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.form = YY.form.apply(lambda x: list(
            set(x).intersection(set(counts.index.values))))
        YY['form_len'] = YY.form.apply(lambda x: len(x))
        # select
        X = XX[YY.form_len > 0]
        Y = YY[YY.form_len > 0]
        mlb.fit(Y.form.values)
        y = mlb.transform(Y.form.values)
    elif ctype == 'rhythm':
        # filter
        counts = pd.Series(np.concatenate(YY.rhythm.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.rhythm = YY.rhythm.apply(lambda x: list(
            set(x).intersection(set(counts.index.values))))
        YY['rhythm_len'] = YY.rhythm.apply(lambda x: len(x))
        # select
        X = XX[YY.rhythm_len > 0]
        Y = YY[YY.rhythm_len > 0]
        mlb.fit(Y.rhythm.values)
        y = mlb.transform(Y.rhythm.values)
    elif ctype == 'all':
        # filter
        counts = pd.Series(np.concatenate(YY.all_scp.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.all_scp = YY.all_scp.apply(lambda x: list(
            set(x).intersection(set(counts.index.values))))
        YY['all_scp_len'] = YY.all_scp.apply(lambda x: len(x))
        # select
        X = XX[YY.all_scp_len > 0]
        Y = YY[YY.all_scp_len > 0]
        mlb.fit(Y.all_scp.values)
        y = mlb.transform(Y.all_scp.values)
    elif ctype == 'mi':
        counts = pd.Series(np.concatenate(YY.mi.values)).value_counts() # 这一步的作用是计算每个标签的数量，np.concatenate()函数的作用是将数据转化为一维数组，pd.Series()函数的作用是将数据转化为Series格式
        counts = counts[counts > min_samples]
        YY.mi = YY.mi.apply(lambda x: list(
            set(x).intersection(set(counts.index.values)))) # 这一步的作用是将标签数据中数量小于min_samples的标签去除，set()函数的作用是去除重复的元素，set(x).intersection(set(counts.index.values))的作用是求x和counts.index.values的交集
        YY['mi_len'] = YY.mi.apply(lambda x: len(x))    # 这一步的作用是计算每个样本的标签数量
        # select
        X = XX[YY.mi_len > 0]
        Y = YY[YY.mi_len > 0]
        mlb.fit(Y.mi.values)
        y = mlb.transform(Y.mi.values)

    return X, Y, y, mlb


def preprocess_signals(X_train, X_validation, X_test, outputfolder):    # 这个函数的作用是对信号数据进行预处理
    # Standardize data such that mean 0 and variance 1
    ss = StandardScaler()   # StandardScaler()函数的作用是对数据进行标准化
    ss.fit(np.vstack(X_train).flatten()[:, np.newaxis].astype(float))   # np.vstack()函数的作用是将数据转化为垂直方向的数组，np.vstack(X_train).flatten()[:, np.newaxis]的作用是将X_train转化为一维数组，astype(float)的作用是将数据转化为float格式，flatten()函数的作用是将数据转化为一维数组

    # Save Standardizer data
    with open(outputfolder+'standard_scaler.pkl', 'wb') as ss_file:
        pickle.dump(ss, ss_file)    # 保存Standardizer数据，pickle.dump()函数的作用是将数据保存为pickle格式

    return apply_standardizer(X_train, ss), apply_standardizer(X_validation, ss), apply_standardizer(X_test, ss)


def apply_standardizer(X, ss):  # 这个函数的作用是对信号数据进行标准化
    X_tmp = []
    for x in X:
        x_shape = x.shape
        X_tmp.append(ss.transform(x.flatten()[:, np.newaxis]).reshape(x_shape))
    X_tmp = np.array(X_tmp)
    return X_tmp
