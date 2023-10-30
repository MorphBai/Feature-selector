import numpy as np
from transformers import BertConfig
from tqdm import tqdm
from argparse import ArgumentParser
from model.model_resnet import ResNet
from model.model_ecgbert import ECGBert
from model.model_ulasnet import UlasNet
from model.model_imelnet import IMELNet
from model.model_seecgnet import SE_ECGNet
from model.model_msnet import MSNet
from model.model_selector import Selector

from utils.data import create_dataloaders
import pandas as pd
import torch
import os
import warnings
from utils.utils import build_optimizer, Logs, seed_everything
import utils.utils as utils
from transformers import BertConfig
from transformers import get_cosine_schedule_with_warmup

import time


def build_model(args, is_train=True):
    if args.model == 'resnet':
        print('Resnet init!')
        model = ResNet(num_classes=5)
    elif args.model == 'seecgnet':
        print('SE_ECGNET init!')
        model = SE_ECGNet(num_classes=5)
    elif args.model == 'ulasnet':
        print('UlasNet init!')
        model = UlasNet(num_classes=5)
    elif args.model == 'imelnet':
        print('IMELNet init!')
        model = IMELNet(num_classes=5)
    elif args.model == 'ecgbert':   #*
        config = BertConfig.from_pretrained(
            args.model_name_or_path, num_labels=5)
        model = ECGBert(config=config)
        print('ECGBert init!')
    elif args.model == 'msnet':
        model = MSNet(num_classes=5)
        print('MSNet init!')
    elif args.model == 'selector':
        config = BertConfig.from_pretrained(
            args.model_name_or_path, num_labels=5)
        model = Selector(config=config)
        print('Post-convolution Feature Selector init!')
    
    else:
        raise NameError('model name incorrect')
    return model


def trainer(model, train_dataloader, optimizer, scheduler, ckpt, args, val_dataloader=None, logs=None):
    early_stopping = args.early_stopping # 此步骤为提前终止训练，当连续10个epoch的验证集准确率没有提升时，停止训练

    best_score = 0  # 保存最好的准确率
    best_dict = {} # 保存最好的准确率对应的epoch
    for epoch in range(args.num_epochs): # 训练epoch
        epoch_loss = 0  # 记录每个epoch的loss
        train_iter = tqdm(
            train_dataloader, desc=f'Epoch:{epoch + 1}', total=len(train_dataloader)) # tqdm是一个进度条库，用于显示训练进度
        model.train() # 训练模式
        torch.cuda.empty_cache() # 清空显存
        for step, inputs in enumerate(train_iter): # 训练step
            inputs = {key: inputs[key].to(args.device)
                      for key in inputs.keys()}
            _, loss = model(**inputs)   # 计算loss
            loss.backward() # 反向传播

            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)  # 梯度裁剪

            optimizer.step()    # 更新参数
            optimizer.zero_grad()   # 清空梯度
            if args.warmup_ratio:   # 学习率衰减，当warmup_ratio为0时，不进行学习率衰减
                scheduler.step()    

            epoch_loss += loss.item()   # 将每个step的loss累加
            train_iter.set_postfix_str(
                f'loss: {epoch_loss / (step+1):.4f}')   # 计算每个epoch的平均loss

        torch.save(model.state_dict(), f'{ckpt}/last_model.pth')    # 保存最后一个epoch的模型

        with torch.no_grad():   # 验证集
            model.eval()    # 验证模式
            eval_loss = 0   # 记录验证集loss
            target = [] # 记录验证集标签
            pred = []   # 记录验证集预测值

            for inputs in val_dataloader:   # 验证集step
                inputs = {key: inputs[key].to(args.device)  # 将数据放入显存
                          for key in inputs.keys()} 
                logits, loss = model(**inputs)  # 计算loss
                target.append(
                    inputs['labels'].detach().cpu().numpy())    # 记录标签
                pred.append(logits.detach().cpu().numpy())  # 记录预测值
                eval_loss += loss.item()    # 记录loss
            eval_dict = utils.evaluate(
                np.concatenate(pred), np.concatenate(target))   # 计算验证集准确率
            eval_dict['loss'] = eval_loss / len(val_dataloader) # 计算验证集平均loss
            logs.write(
                f'=============== epoch {epoch+1} ===============\n')   # 将每个epoch的结果写入log文件
            for k, v in eval_dict.items():  # 打印每个epoch的结果
                print(f'{k}: {v}')
                logs.write(f'{k}: {v}\n')

            if eval_dict['accuracy'] > best_score:  # 保存最好的模型
                early_stopping = args.early_stopping    # 重置early_stopping
                best_score = eval_dict['accuracy']  # 更新最好的准确率
                torch.save(model.state_dict(), f'{ckpt}/best_model.pth')    # 保存最好的模型
                best_dict = eval_dict   # 保存最好的模型对应的结果
            else:
                early_stopping -= 1 # early_stopping减1，early_stopping的作用是当连续10个epoch的验证集准确率没有提升时，停止训练
                if early_stopping == 0: # 当early_stopping为0时，停止训练
                    print(best_dict)    # 打印最好的模型对应的结果
                    return  # 停止训练
    print(best_dict)


def train(args):    # 训练函数
    train_dataloader, val_dataloader = create_dataloaders(args) # 创建训练集和验证集的dataloader
    model = build_model(args)   # 创建模型
    model.to(args.device)   # 将模型放入显存
    optimizer = build_optimizer(model, args)    # 创建优化器
    if args.warmup_ratio:   # 学习率衰减，当warmup_ratio为0时，不进行学习率衰减
        train_steps = args.num_epochs * len(train_dataloader)   # 计算训练总step
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_training_steps=train_steps * args.warmup_ratio,
            num_warmup_steps=train_steps)   # 创建学习率衰减器
    else:
        scheduler = None    # 不进行学习率衰减

    ckpt = os.path.join(
        args.checkpoint, args.model_type)
    os.makedirs(ckpt, exist_ok=True)    # 创建checkpoint文件夹
    logs = Logs(os.path.join(ckpt, 'log.txt'))  # 创建log文件
    for k, v in vars(args).items(): # 将参数写入log文件
        logs.write(f'{k}: {v}' + '\n')

    trainer(model,
            train_dataloader,
            optimizer,
            scheduler,
            ckpt,
            args,
            val_dataloader=val_dataloader,
            logs=logs)  # 开始训练


def evaluate(args):
    root_time = time.time()
    dataloader = create_dataloaders(args, train_mode=False) # 创建测试集dataloader

    model = build_model(args, is_train=False)   # 创建模型
    model.load_state_dict(torch.load(args.eval_model, map_location='cuda'))  # 加载模型
    model.to(args.device)   # 将模型放入显存
    model.eval()    # 测试模式

    target = [] # 记录标签
    pred = []   # 记录预测值
    eval_loss = 0   # 记录loss
    with torch.no_grad():   # 测试集
        start_time = time.time()
        for inputs in tqdm(dataloader): # 测试集step
            inputs = {key: inputs[key].to(args.device)
                      for key in inputs.keys()} # 将数据放入显存
            logits, loss = model(**inputs)  # 计算loss
            target.append(
                inputs['labels'].detach().cpu().numpy())    # 记录标签
            pred.append(logits.detach().cpu().numpy())  # 记录预测值
            eval_loss += loss.item()    # 记录loss
        end_time = time.time()
        eval_dict = utils.evaluate(
            np.concatenate(pred), np.concatenate(target), eval_mode=True)   # 计算测试集准确率
        np.save('pred.npy', np.concatenate(pred))   # 保存预测值
        np.save('target.npy', np.concatenate(target))   # 保存标签
        eval_dict['loss'] = eval_loss / len(dataloader) # 计算测试集平均loss
        eval_dict['prepare_time'] = start_time - root_time
        eval_dict['time'] = end_time - start_time
        for k, v in eval_dict.items():  # 打印测试集结果
            print(f'{k}: {v}')


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--model_type', type=str,
                        default='demo') # model_type: demo, base, large * <----------
    parser.add_argument('--model', type=str, required=True) # models: resnet, seecgnet, ulasnet, imelnet, ecgbert, msnet * <----------
    # parser.add_argument('--model', type=str, default='ecgbert')
    parser.add_argument('--model_name_or_path', type=str,
                        default='./model')   # bert预训练模型路径
    parser.add_argument('--task', type=str, default='mi')   # task: mi, norm * <----------

    parser.add_argument('--checkpoint', type=str,
                        default='./checkpoint')   # checkpoint路径

    parser.add_argument('--num_workers', type=int, default=4)   # dataloader的num_workers
    parser.add_argument('--batch_size', type=int, default=16) # msnet 32
    parser.add_argument('--val_batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=64)

    parser.add_argument('--do_eval', action='store_true', default=False)    # 是否进行测试，action='store_true'表示如果有这个参数，则为True，否则为False * <----------
    parser.add_argument('--eval_model', type=str)   # 测试模型路径，checkpoint路径下的模型 * <----------

    parser.add_argument('--lr', type=float, default=5e-6)   # 学习率 * <----------
    parser.add_argument('--bert_lr', type=float, default=5e-5)  # bert学习率（无效）
    parser.add_argument('--clf_lr', type=float, default=5e-5)   # 分类器学习率（无效）
    parser.add_argument('--weight_decay', type=float, default=1e-4)  # 权重衰减

    parser.add_argument('--num_epochs', type=int, default=100)  # 训练epoch
    parser.add_argument('--early_stopping', type=int, default=10)   # 提前终止训练

    parser.add_argument('--warmup_ratio', type=float, default=0)    # 学习率衰减比例，0~0.99

    parser.add_argument('--device', type=str, default='cuda')    # device: cuda or cpu
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    warnings.filterwarnings("ignore")   # 忽略警告

    seed_everything(args.seed)  # 设置随机种子

    for k, v in vars(args).items(): # 打印参数
        print(f'{k}: {v}')

    if args.do_eval:    # 测试模式，如果do_eval为True，则进行测试，否则进行训练
        evaluate(args)
    else:
        train(args)
