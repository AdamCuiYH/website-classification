import pandas as pd
import time
import multiprocessing
from utils import chrome_spyder
from simpletransformers.classification import ClassificationModel
global html_content

if __name__ == '__main__':
    
    print('          ----------   开始爬虫,请耐心等待   ----------')
    # 调用爬虫模块，得到爬虫文本
    start1 = time.time()
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    with open('url.txt', "r", encoding='utf-8') as f:
        C = f.read()
        urls =C.split()
        #urlss = urls[30:]
        print('一共爬虫网站数量：',len(urls))
    pool.map(chrome_spyder, urls)
    pool.close()
    pool.join()
    end1 = time.time()
    print('          ------------   爬虫结束   -------------')
    print(end1-start1)
    

    # 读取爬虫的文本，转成 DataFrame 的形式
    data_df = pd.read_csv('爬虫文本.csv', header=None, usecols=[0, 1, 2], names=['status', 'url', 'content'], dtype='str')
    df1 = data_df.dropna(subset=['content'])
    df2 = df1.loc[df1['status'] == 'True']
    data_df1 = df2[['url','content']]
    data_df2 = data_df1.copy()
    df4 = list(data_df2['content'])
    
    print('           -------------   开始识别网站   -------------')
    
    # 模型预测
    model = ClassificationModel('bert', 'model_weight',use_cuda=False,args={"thread_count":1})
    start = time.time()
    predictions, raw_outputs = model.predict(df4)
    end = time.time()
    print(end-start,'s')
    dict_prediction = {0: '贷款-P2P', 1: '贷款-抵押', 2: '贷款-小额', 3: '贷款-咨讯', 4: '贷款-综合', 5: '贷款-租赁', 6: '赌-彩票预测',
						7: '赌-赌场系', 8: '赌-购彩系', 9: '赌-电子游戏', 10: '赌-球', 11: '黄-视频', 12: '黄-成人用品药', 13: '签名网站',
					   14: '黄-小说漫画', 15: '黄-性感图', 16: '黄-直播', 17: '宗教-场所', 18: '宗教-机构', 19: '宗教-文化', 20: '宗教-用品',
					   21: 'vpn-非法', 22: 'vpn-商务', 23: '打码', 24: 'VPS', 25: '短链接', 26: '配资', 27: '其他', 28: '四方支付',
					   29: '云发卡', 30: '流量刷单', 31: '微交易', 32: '云呼'}
    predictions_label = [dict_prediction[i] if i in dict_prediction else i for i in predictions]
    data_df2['BERT预测标签'] = predictions_label
    df5 = data_df.loc[(data_df['status'] == 'False') | (data_df['content'].isnull())]
    df6 = df5['url']

    # 将预测结果导出
    writer = pd.ExcelWriter('网站识别结果.xlsx', engine='xlsxwriter',
                            options={'strings_to_urls': False})
    data_df2.to_excel(writer, index=None,sheet_name='网站类别')
    df6.to_excel(writer,sheet_name='爬虫失败和内容为空的网址')
    writer.close()

    # 统计所用时间
    end2 = time.time()

    print('          ----------------   程序运行结束，已退出   ---------------')
    print('网页爬虫用时：', (end1 - start1), 's')
    print('模型预测用时', (end2 - end1), 's')
    print('一共用时', (end2 - start1), 's')
    