import pandas as pd
import time
from utils import chrome_spyder
from simpletransformers.classification import ClassificationModel
global html_content

if __name__ == '__main__':
    '''
    print('          ------------------------------   开始爬虫,请耐心等待   ------------------------------')
    # 调用爬虫模块，得到爬虫文本
    start1 = time.time()
    with open('url.txt', "r", encoding='utf-8') as f:
        C = f.read()
        urls =C.split()
        #urlss = urls[30:]
        print('一共爬虫网站数量：',len(urls))
        for url in urls:
            chrome_spyder(url)
    end1 = time.time()
    print('          --------------------------------   爬虫结束   ---------------------------------')
    print(end1-start1)
    '''

    # 读取爬虫的文本，转成 DataFrame 的形式
    data_df = pd.read_csv('爬虫文本.csv', header=None, usecols=[0, 1, 2], names=['status', 'url', 'content'], dtype='str')
    print(data_df)
    df1 = data_df.loc[data_df['status'] == 'True']
    print(df1)
    df2 = df1.dropna(subset=['content'])
    print(df2)
    
    # df3 得到过滤这些关键词的文本
    # df3 = df2[~df2['content'].str.contains(
    #     '401 authorization|errorget|error|服务器错误|400 unknow|400 bad request|an error|not found|NotFound|501 not implemented|410 gone|网站不存在|500 internal|InvalidURLTherequestedURL|NotFound|errorcode|403 forbidden|invalid url|nosuchbucket|404 not found not|Sorry,thewebisForbidden|errorcode|501 not|502 bad|503 service|accessdenied|ERROR|页面不存|系统发生错误')]
    # df3 = df3.reset_index()
    data_df1 = df2[['url','content']]
    data_df2 = data_df1.copy()
    df4 = list(data_df2['content'])
    
    print('           ---------------------------------   开始识别网站   ---------------------------------')
    
    # 模型预测
    model = ClassificationModel('bert', r'F:\train_data\9.18\outputs919更好', use_cuda=False)
    start = time.time()
    predictions, raw_outputs = model.predict(df4)
    end = time.time()
    print(end-start,'s')
    dict_prediction = {0: '涉贷款', 1: '涉赌', 2: '涉黄', 3: '宗教', 4: 'VPN', 5: '正常', 6: '打码', 7: '正常', 8: '短链接', 9: '配资'}
    predictions_label = [dict_prediction[i] if i in dict_prediction else i for i in predictions]
    print(predictions_label)
    
    data_df2['BERT预测标签'] = predictions_label

    df5 = data_df.loc[data_df['status'] == 'False']
    df6 = data_df.loc[(data_df['status'] == 'True') & (data_df['content'].isnull())]
    df7 = df5['url']
    df8 = df6['url']

    # 将预测结果导出
    writer = pd.ExcelWriter('网站分类预测结果.xlsx', engine='xlsxwriter',
                            options={'strings_to_urls': False})
    data_df2.to_excel(writer, index=None,sheet_name='网站类别')
    df7.to_excel(writer,sheet_name='爬虫失败的网址')
    df8.to_excel(writer,sheet_name='爬虫内容为空的网址')
    writer.close()

    # 统计所用时间
    end2 = time.time()

    print('          -------------------------------   程序运行结束，已退出   ------------------------------')
    print('网页爬虫用时：', (end1 - start1), 's')
    print('模型预测用时: ', (end2 - end1), 's')
    print('一共用时: ', (end2 - start1), 's')
    