import datetime
import glob
import os
import random

import numpy as np
import pandas as pd
import streamlit as st
import torch
import yfinance as yf
from torch import nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torch.utils.data import Dataset, DataLoader

divide_class = ('train', 'valid', 'test')
input_days = 30
folder_suffix, stock_fn = '.stp', 'stock.csv'
divide_fn, network_fn = 'divide.npz', 'network.pth'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_stock(stock_code, start_data, end_date):
    # 从Yahoo Finance获取股票历史数据
    stock_data = yf.download(stock_code, start=start_data, end=end_date)

    if stock_code.find("=") != -1:
        return stock_data.drop(columns=['Volume'])
    else:
        return stock_data


def gen_multi_col(stock_code, old_col_list):
    # 合并多重索引
    return [f'{stock_code}-{old_col}' for old_col in old_col_list]


def get_stock_list(data_set_stock_code='000725.SZ', start_data='2020-05-10', end_date='2023-05-10'):
    # 道琼斯、标普500、上证指数、沪深300、人民币/美元汇率、人民币/港币汇率
    base_stock_code = [data_set_stock_code, '^DJI', '^GSPC', '000001.SS', '000300.SS', 'CNY=X', 'HKDCNY=X']

    stock_data_list, multi_col_list = [], []
    for stock_code in base_stock_code:
        temp_stock_df = get_stock(stock_code, start_data, end_date)
        stock_data_list.append(temp_stock_df)
        multi_col_list.extend(gen_multi_col(stock_code, temp_stock_df.columns))

    merged_df = pd.concat(stock_data_list, axis=1)
    merged_df.fillna(0, inplace=True)
    merged_df.columns = multi_col_list

    bool_series = merged_df[f'{data_set_stock_code}-Close'] != 0
    merged_df = merged_df.loc[bool_series]

    if merged_df.empty:
        st.error('未找到对应股票在选择日期下的数据')
        return False

    else:
        folder_path = f'{data_set_stock_code}_{start_data}-{end_date}{folder_suffix}'
        save_df_path = f'{folder_path}/stock.csv'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        merged_df.to_csv(save_df_path)
        return merged_df, base_stock_code, save_df_path


def read_stock_csv(stock_file_name):
    read_dataset = pd.read_csv(stock_file_name, index_col=0)
    stock_code = stock_file_name.split('_')[0]
    return read_dataset, stock_code


def random_select(stock_file_name, train_percent=0.8, cross_valid_percent=0.25):
    read_dataset, stock_code = read_stock_csv(stock_file_name)
    num_files = len(read_dataset) - (input_days + 1)

    all_index_list = list(range(num_files))
    spilt_point = int(train_percent * num_files)

    train, test = all_index_list[:spilt_point], all_index_list[spilt_point:]
    cross_valid = sorted(random.sample(train, int(spilt_point * cross_valid_percent)))

    single_train_path = stock_file_name.replace(stock_fn, f'train_{train_percent}-{cross_valid_percent}')
    if not os.path.exists(single_train_path):
        # 如果目录不存在，则创建目录
        os.makedirs(single_train_path)

    divide_file_name = os.path.join(single_train_path, divide_fn)

    save_dict = {divide_class[0]: np.array(train),
                 divide_class[1]: np.array(cross_valid),
                 divide_class[2]: np.array(test)}
    np.savez(divide_file_name, **save_dict)

    divide_str = f'{divide_class[0]}: {len(train)} - ' \
                 f'{divide_class[1]}: {len(cross_valid)} - ' \
                 f'{divide_class[2]}: {len(test)} - All file num: {num_files}'

    return divide_str, divide_file_name


class StockDataset(Dataset):
    def __init__(self, divide_file_name, stock_file_name, data_class):
        self.read_dataset, self.stock_code = read_stock_csv(stock_file_name)
        self.data_class_index = np.load(divide_file_name)[data_class]

    def __len__(self):
        return len(self.data_class_index)

    def __getitem__(self, idx):
        select_data_idx = self.data_class_index[idx]

        co_days = self.read_dataset[select_data_idx + input_days:select_data_idx + input_days + 1].index.values[0]

        input_stock = torch.tensor(self.read_dataset[select_data_idx:select_data_idx + input_days].values,
                                   dtype=torch.float32)

        target_close = torch.tensor(self.read_dataset[f'{self.stock_code}-Close']
                                    [select_data_idx + input_days:select_data_idx + input_days + 1].values,
                                    dtype=torch.float32)

        return input_stock, target_close, co_days


class StockTransformer(nn.Module):
    def __init__(self, stock_days=30, stock_dim=40, embed_dim=256,
                 n_head=16, d_hid=2048, net_layers=10, dropout=0.1):
        super().__init__()

        # PTF encoder specifics-----------------------------------------------------
        self.first_norm = nn.LayerNorm(stock_dim)
        self.stock_embed = nn.Linear(stock_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, stock_days + 1, embed_dim))

        encoder_layers = TransformerEncoderLayer(embed_dim, n_head, d_hid, dropout, batch_first=True)
        self.phase_encoder = TransformerEncoder(encoder_layers, net_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Linear(embed_dim, embed_dim // 2)
        self.activate = nn.SELU()
        self.prediction = nn.Linear(embed_dim // 2, 1)

        # --------------------------------------------------------------------------

    def forward(self, src):
        # normalize input
        src = self.first_norm(src)

        # embed phase
        src = self.stock_embed(src)

        # add pos embed without cls token
        src = src + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(src.shape[0], -1, -1)
        src = torch.cat((cls_tokens, src), dim=1)

        # apply Transformer blocks
        src = self.phase_encoder(src)
        src = self.norm(src)
        src = self.mlp(src[:, 0, :])
        src = self.activate(src)
        src = self.prediction(src)

        return src


def streamlit_tab1_info(stock_df, stock_code_list):
    with st.expander('数据格式详情介绍'):
        st.write('数据采集阶段，本文收集道琼斯、标普500、上证指数、沪深300、人民币/美元汇率、人民币/港币汇率作为基本面数据。'
                 '随后附加上述所选框中的对应股票，共同用作数据集。所选用股票代码如下所示：')
        st.write(stock_code_list)
        st.write('收集数据中NAN默认补0，数据集以DataFrame格式展示如下：')
        st.dataframe(stock_df)


def streamlit_tab1():
    with st.form("select_stock_info"):
        col1, col2, col3 = st.columns(3)
        with col1:
            input_stock_code = st.text_input("请输入股票代码", "000725.SZ")
        with col2:
            start_day = st.date_input("选择股票起始时间", datetime.date(2022, 5, 10))
        with col3:
            end_day = st.date_input("选择股票结束时间", datetime.date(2023, 5, 10))

        st.info('数据来源为雅虎财经，此处默认股票数据集为A股京东方，股票代码：000725.SZ')

        submitted = st.form_submit_button("Submit")

    if submitted:
        with st.spinner('等待雅虎财经返回数据...'):
            pack_return = get_stock_list(input_stock_code, start_day, end_day)

            if pack_return:
                stock_df, all_stock_code, save_df_path = pack_return[0], pack_return[1], pack_return[2]
                st.success(f"选择股票：{input_stock_code} / 开始时间：{start_day} / 结束时间：{end_day}  \n"
                           f"股票数据保存至：{save_df_path}")
                streamlit_tab1_info(stock_df, all_stock_code)


def clear_memory():
    if device == 'cuda:0':
        for i in range(10):
            torch.cuda.empty_cache()


def streamlit_tab2_train(divide_file_name, stock_file_name, data_class=divide_class[0],
                         net_layer=10, lr=5e-5, epoch=1000, batch_size=64, shuffle=True):
    stock_dataset = StockDataset(divide_file_name, stock_file_name, data_class)
    stock_dataloader = DataLoader(stock_dataset, batch_size, shuffle)

    net_save_path = divide_file_name.replace(divide_fn, f'{net_layer}l-{network_fn}')
    stn = StockTransformer(net_layers=net_layer)
    stn = stn.to(device)
    stn.train()

    opt = torch.optim.Adam(stn.parameters(), lr=lr)  # 优化器
    loss_fun = nn.MSELoss()  # recommended lr: 5e-5, 3e-5, 2e-5

    progress_text = st.write(f"net_layers={net_layer} / lr={lr} / epoch={epoch} / batch_size={batch_size}")
    my_bar = st.progress(0.0, text=progress_text)

    for each_epoch in range(epoch):

        for x, y, _ in stock_dataloader:
            inputs = x.to(device)
            targets = y.to(device)
            opt.zero_grad()  # 梯度归零

            outputs_ecg = stn(inputs)  # 前向传播
            loss = loss_fun(outputs_ecg, targets)  # 计算损失

            loss.backward()  # 反向传播
            opt.step()  # 梯度下降

            progress_text = f'epoch {each_epoch + 1:2d}/{epoch} - mse_loss: {loss.item():.3f}'
            my_bar.progress(each_epoch / epoch, text=progress_text)

    torch.save(obj=stn.state_dict(), f=net_save_path)
    my_bar.progress(1.0, text=f"ViT网络训练完成 模型文件保存至：{net_save_path}")


def streamlit_tab2_info(stock_file_name):
    read_dataset, stock_code = read_stock_csv(stock_file_name)
    targets_col = f'{stock_code}-Close'

    with st.expander('数据格式详情介绍'):
        # 展示训练数据，读取数据集前30行作为训练集，第31行作为预测目标
        if targets_col in read_dataset.columns:
            st.write(f'该数据选取{targets_col}作为预测目标，使用T-29~T预测T+1的收盘价')
            st.write(f'数据集前30行作为训练集，{stock_code}对应收盘价的第31行作为预测目标')
            col1, col2 = st.columns([5, 3])
            col1.dataframe(read_dataset[0:30])
            col2.dataframe(read_dataset[targets_col][30:31])


def streamlit_tab2():
    with st.form("read_stock_dataset"):
        # 获取当前目录下*.stp文件夹路径
        stock_folder_list = glob.glob(f'*{folder_suffix}')
        stock_folder_name = st.selectbox('读取股票数据集：', stock_folder_list)

        net_col1, net_col2, net_col3 = st.columns(3)
        with net_col1:
            train_percent = st.number_input('选择训练集比例：', 0.1, 0.9, 0.8)
            lr_select_list = [5e-5, 3e-5, 2e-5]
            learning_rate = st.selectbox('选择学习率：', lr_select_list, index=0)
        with net_col2:
            cross_valid_percent = st.number_input('选择交叉验证比例：', 0.1, 0.9, 0.25)
            epoch_num = st.number_input('选择训练轮数：', 1, 8000, 4)
        with net_col3:
            net_layer_num = st.number_input('选择深度网络层数：', 1, 18, 4)
            batch_size_num = st.number_input('选择批次大小：', 1, 2048, 100)

        dis_submit = False if stock_folder_list else True
        submitted = st.form_submit_button("Submit", disabled=dis_submit)

    if submitted:
        clear_memory()  # 清理显存
        stock_file_name = os.path.join(stock_folder_name, 'stock.csv')
        divide_str, divide_file_name = random_select(stock_file_name, train_percent, cross_valid_percent)
        st.success(f'分割索引保存至：{divide_file_name}  \n数据分割情况为：{divide_str}')
        streamlit_tab2_info(stock_file_name)
        with st.expander("深度网络训练：", expanded=True):
            streamlit_tab2_train(divide_file_name, stock_file_name, net_layer=net_layer_num,
                                 lr=learning_rate, epoch=epoch_num, batch_size=batch_size_num)


def check_path_join(check_folder, base_path):
    if check_folder and base_path:
        join_path = os.path.join(check_folder, base_path)
    else:
        join_path = None
    return join_path


def check_list_exist(check_folder, base_list_name):
    if check_folder and base_list_name:
        base_list = glob.glob(os.path.join(check_folder, base_list_name))
        base_list = [os.path.basename(each_base) for each_base in base_list]
    else:
        base_list = []

    return base_list


def streamlit_tab3_predict(divide_file_name, stock_file_name, net_load_path,
                           data_class=divide_class[2], batch_size=64, shuffle=False):
    stock_dataset = StockDataset(divide_file_name, stock_file_name, data_class)
    stock_dataloader = DataLoader(stock_dataset, batch_size, shuffle)

    net_layer = int(os.path.basename(net_load_path).split('l-')[0])
    stn = StockTransformer(net_layers=net_layer)
    stn.load_state_dict(torch.load(net_load_path, map_location=device))
    stn.to(device)
    stn.eval()

    loss_fun = nn.MSELoss()  # recommended lr: 5e-5, 3e-5, 2e-5

    progress_view_text = f"net-layers={net_layer} / batch-size={batch_size}"
    my_bar = st.progress(0.0, text=progress_view_text)

    target_list, output_list, co_days_list = [], [], []

    for i, (x, y, z) in enumerate(stock_dataloader):
        inputs = x.to(device)
        targets = y.to(device)
        outputs = stn(inputs)  # 前向传播

        target_list.append(targets.detach().cpu())
        output_list.append(outputs.detach().cpu())
        co_days_list.extend(list(z))

        progress_text = f'batch {i + 1:2d}/{len(stock_dataloader)}'
        my_bar.progress((i + 1) / len(stock_dataloader), text=progress_text)

    target_list = torch.cat(target_list, dim=0)
    output_list = torch.cat(output_list, dim=0)

    st.success(f'{progress_view_text}  \n'
               f'ViT网络预测完成 / MSE误差为：{loss_fun(target_list, output_list).item():.4f}')
    my_bar.empty()
    return target_list.numpy().flatten(), output_list.numpy().flatten(), np.array(co_days_list).flatten()


def streamlit_tab3_submitted(stock_folder, train_path, select_network, select_dataset, select_batch):
    clear_memory()  # 清理显存

    divide_file_path = os.path.join(stock_folder, train_path, divide_fn)
    stock_file_path = os.path.join(stock_folder, stock_fn)
    network_file = os.path.join(stock_folder, train_path, select_network)

    target, output, date = streamlit_tab3_predict(divide_file_path, stock_file_path, network_file,
                                                  data_class=select_dataset, batch_size=select_batch)
    predict_df = pd.DataFrame({'target': target, 'output': output}, index=date)
    with st.expander("股票时序预测结果：", expanded=True):
        st.line_chart(predict_df)

    predict_df.to_csv(network_file.replace(network_fn, f'{select_dataset}.csv'))


def streamlit_tab3():
    with st.expander("stock_prediction", True):
        col1, col2 = st.columns([2, 1])
        with col1:
            # 获取当前目录下stock*.csv文件的路径
            stock_folder = st.selectbox('读取股票数据集：', glob.glob(f'*{folder_suffix}'))
            train_list = check_list_exist(stock_folder, '*train*')

            train_col, net_col = st.columns(2)
            with train_col:
                select_train = st.selectbox('选择股票对应的一次训练：', train_list)

            with net_col:
                network_list = check_list_exist(stock_folder, check_path_join(select_train, f'*{network_fn}'))
                select_network = st.selectbox('选择股票对应神经网络：', network_list)

        with col2:
            select_dataset = st.selectbox('选择分割数据集：', divide_class, index=2)
            select_batch = st.number_input('选择批次大小：', 1, 1024, 512)

        # 格式化文件路径

        sub_visible = False if select_network else True
        sub_bt = st.button("Submit", disabled=sub_visible, key='tab3_sub')

    if sub_bt:
        streamlit_tab3_submitted(stock_folder, select_train, select_network, select_dataset, select_batch)


def streamlit_tab4_info(pred_file_name, ori_money):
    stock_code = pred_file_name.split('_')[0]

    st.success(f"使用下面情况对预测数据进行回测：  \n"
               f"选择股票：{stock_code} / 初始本金：{ori_money}")


def streamlit_tab4_invest(pred_file_name, ori_money, capital_percent=0.5, share_percent=0.5):
    # 获取股票价格数据
    stock_data = pd.read_csv(pred_file_name, index_col=0)

    # 初始化本金和股票数量
    num_capital, value_list = ori_money, []
    today_price, num_shares = 0, 0

    # 遍历每一天的数据
    for i in range(len(stock_data) - 1):
        # 获取当天的股票价格
        today_price, tomorrow_pred = stock_data['target'][i], stock_data['output'][i + 1]

        # 如果神经网络预测的收盘价高于当前股票价格，则买入该股票
        if tomorrow_pred > today_price:
            buy_share = int(capital_percent * num_capital) // today_price
            num_shares += buy_share
            num_capital -= buy_share * today_price

        # 如果神经网络预测的收盘价低于当前股票价格，则卖出该股票
        elif tomorrow_pred < today_price:
            sell_share = int(share_percent * num_shares)
            num_capital += sell_share * today_price
            num_shares -= sell_share

        value_list.append(num_capital + num_shares * today_price)

    value_list.append(num_capital + num_shares * stock_data['target'][-1])

    with st.expander("股票投资回测结果：", expanded=True):
        mse_loss = stock_data.diff(axis=1).dropna(axis=1).apply(lambda x: x ** 2).mean().values[0]
        st.write(f'Capital: {num_capital:.2f} / Shares: {num_shares:.2f} / Mse loss: {mse_loss:.4f}')
        st.line_chart(stock_data)

        # 计算并输出最终的资产价值
        final_value = num_capital + num_shares * today_price
        st.write(f'最终资产: {final_value:.2f} / 收益: {final_value - ori_money:.2f} / '
                 f'收益率：{(final_value - ori_money) / ori_money * 100:.2f}%')

        value_df = pd.DataFrame(np.array(value_list), index=stock_data.index, columns=['all_value'])
        st.line_chart(value_df)


def streamlit_tab4():
    with st.expander("stock_trade", True):
        col1, col2 = st.columns([6, 4])
        with col1:
            stock_folder = st.selectbox('选择回测股票：', glob.glob(f'*{folder_suffix}'))
            train_list = check_list_exist(stock_folder, '*train*')
            select_train = st.selectbox('选择回测股票的一次训练：', train_list)

        with col2:
            pred_list = check_list_exist(stock_folder, check_path_join(select_train, '*.csv'))
            pred_for_backtest = st.selectbox('选择回测股票的对应数据集：', pred_list)
            ori_money = st.number_input('初始资金：', 10000, 1000000, 100000)

        sub_visible = False if pred_for_backtest else True
        submitted = st.button("Submit", disabled=sub_visible, key='tab4_sub')

    if submitted:
        match_pred_test = os.path.join(stock_folder, select_train, pred_for_backtest)
        streamlit_tab4_info(stock_folder, ori_money)
        streamlit_tab4_invest(match_pred_test, ori_money)


def streamlit_page():
    tab_list = ["股票数据收集", "股票模型训练", "股票数据预测", "股票交易回测"]
    tab1, tab2, tab3, tab4 = st.tabs(tab_list)

    with tab1:
        st.header(tab_list[0])
        streamlit_tab1()

    with tab2:
        st.header(tab_list[1])
        streamlit_tab2()

    with tab3:
        st.header(tab_list[2])
        streamlit_tab3()

    with tab4:
        st.header(tab_list[3])
        streamlit_tab4()


if __name__ == '__main__':
    streamlit_page()
